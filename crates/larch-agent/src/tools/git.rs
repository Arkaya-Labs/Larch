//! `changed_files`, `diff_file`, `show_file` and `log`: the `git` binary run in the first allowed
//! folder, with a fixed environment and no access to the host's git configuration.

use std::{
    ffi::OsString,
    fs::{self, File},
    io::{self, Read},
    path::Path,
    process::{Command, Stdio},
};

use serde_json::{Value, json};

use crate::{
    tool::{
        Builtin, Permission, Tool, ToolCtx, ToolError, first_folder, plain, str_arg, whole_arg,
    },
    tools::files::MAX_READ,
};

pub(crate) fn tools() -> Vec<Box<dyn Tool>> {
    let text = |description: &str| json!({"type": "string", "description": description});
    // Every argument but `base` is required.
    let tool = |name, description, max_result_tokens, properties: Value, call| {
        let required: Vec<&String> = properties
            .as_object()
            .into_iter()
            .flat_map(|p| p.keys())
            .filter(|k| *k != "base")
            .collect();
        let schema = json!({"type": "object", "properties": properties, "required": required});
        Builtin::boxed(
            name,
            description,
            Permission::Read,
            max_result_tokens,
            schema,
            call,
        )
    };
    let branch = text("Branch to inspect");
    let base = text("Branch or commit to compare against; HEAD when left out");
    let path = text("File path inside the repository");
    vec![
        tool(
            "changed_files",
            "List the files a branch changed since it left base.",
            500,
            json!({"branch": branch, "base": base}),
            changed_files,
        ),
        tool(
            "diff_file",
            "Show one file's diff on a branch since it left base.",
            2000,
            json!({"branch": branch, "path": path, "base": base}),
            diff_file,
        ),
        tool(
            "show_file",
            "Show a file as it is at a branch, tag or commit.",
            2000,
            json!({"ref": text("Branch, tag or commit"), "path": path}),
            show_file,
        ),
        tool(
            "log",
            "List a branch's latest n commits, newest first.",
            500,
            json!({"branch": branch, "n": {"type": "integer", "minimum": 1}}),
            log,
        ),
    ]
}

/// A branch, ref or base argument; a leading `-` would make git read it as an option.
fn rev<'a>(args: &'a Value, key: &str) -> Result<&'a str, ToolError> {
    let rev = match key {
        "base" => args[key].as_str().unwrap_or("HEAD"),
        _ => str_arg(args, key)?,
    };
    if rev.starts_with('-') {
        return Err(ToolError::Invalid(format!(
            "{key} '{rev}' must not start with '-'"
        )));
    }
    Ok(rev)
}

/// `args` run by git on the first allowed folder's own repository.
fn git(ctx: &ToolCtx, args: &[&str]) -> Result<String, ToolError> {
    let first = ctx.allowed.first().ok_or(ToolError::Denied(".".into()))?;
    own(&first_folder(first)?, args)?.ok_or_else(|| {
        ToolError::Invalid(format!(
            "'{}' is not a git repository; the git tools read only the first allowed folder's own",
            first.display()
        ))
    })
}

/// `args` run by git on the repository whose top is the canonical folder `real`; `None` when
/// `real` is not the top of one.
pub(crate) fn own(real: &Path, args: &[&str]) -> Result<Option<String>, ToolError> {
    let repo = plain(real.to_path_buf());
    // GIT_DIR skips git's check that the repository belongs to this user, which only a search
    // makes; so git searches once, refusing another user's repository on every system, and must
    // find the folder itself, not one around it.
    let top = run(&repo, None, &["rev-parse", "--show-toplevel"])?;
    if fs::canonicalize(top.trim_end_matches(['\r', '\n']))
        .ok()
        .as_deref()
        != Some(real)
    {
        return Ok(None);
    }
    run(&repo, Some(&repo.join(".git")), args).map(Some)
}

/// git run in `repo` with a fixed environment, on `git_dir` or, without it, on the repository a
/// search from `repo` finds.
pub(crate) fn run(repo: &Path, git_dir: Option<&Path>, args: &[&str]) -> Result<String, ToolError> {
    run_env(repo, git_dir, &[], args)
}

/// `run` with `env` set besides: a work tree or an index of the caller's own.
pub(crate) fn run_env(
    repo: &Path,
    git_dir: Option<&Path>,
    env: &[(&str, &Path)],
    args: &[&str],
) -> Result<String, ToolError> {
    let io = |e| ToolError::Io("git".into(), e);
    let home = tempfile::tempdir().map_err(io)?;
    // stderr goes to a file, so git never blocks on it while stdout is read only in part.
    let stderr = home.path().join("stderr");
    let mut command = Command::new("git");
    crate::tool::no_window(&mut command)
        .env_clear()
        .envs(std::env::var_os("PATH").map(|p| ("PATH", p)))
        .env("HOME", home.path())
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("LC_ALL", "C");
    // The command itself never searches: from a folder that is not a repository it would climb
    // into the one around it, and a ceiling cannot stop that, because GIT_CEILING_DIRECTORIES
    // splits at ':' (';' on Windows), which a folder name may hold.
    if let Some(git_dir) = git_dir {
        command.env("GIT_DIR", git_dir);
    }
    command.envs(env.iter().copied());
    // No hooks: a worktree, commit or merge would run the repository's scripts; the empty
    // home holds none.
    let mut hooks = OsString::from("core.hooksPath=");
    hooks.push(home.path());
    let mut child = command
        // No network: a partial clone would fetch a missing object from its server, and git
        // before 2.44 ignores GIT_NO_LAZY_FETCH, so every transport is refused as well.
        .env("GIT_NO_LAZY_FETCH", "1")
        .env("GIT_TERMINAL_PROMPT", "0")
        .args(["-c", "protocol.allow=never"])
        // A signed commit would make log run the program the repository's gpg.program names.
        .args(["-c", "log.showSignature=false"])
        .arg("-c")
        .arg(hooks)
        .current_dir(repo)
        .args(["-c", "core.quotePath=false"])
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(File::create(&stderr).map_err(io)?)
        .spawn()
        .map_err(io)?;
    let mut out = Vec::new();
    let read = read_capped(child.stdout.take().expect("stdout is piped"), &mut out);
    let over = out.len() as u64 > MAX_READ;
    if over || read.is_err() {
        child.kill().map_err(io)?;
    }
    let status = child.wait().map_err(io)?;
    read.map_err(io)?;
    if over {
        return Err(ToolError::Invalid(format!(
            "git printed more than {MAX_READ} bytes; the git tools read up to {MAX_READ} bytes"
        )));
    }
    if !status.success() {
        let stderr = fs::read(&stderr).map_err(io)?;
        return Err(ToolError::Git(
            String::from_utf8_lossy(&stderr).trim().into(),
        ));
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// At most `MAX_READ + 1` bytes of `from` into `out`: one byte more shows the output was too long,
/// and git's output is never loaded past it.
fn read_capped(from: impl Read, out: &mut Vec<u8>) -> io::Result<usize> {
    from.take(MAX_READ + 1).read_to_end(out)
}

fn changed_files(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let range = format!("{}...{}", rev(args, "base")?, rev(args, "branch")?);
    let out = git(ctx, &["diff", "--no-ext-diff", "--name-only", &range, "--"])?;
    Ok(out.trim_end().into())
}

fn diff_file(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let range = format!("{}...{}", rev(args, "base")?, rev(args, "branch")?);
    let path = str_arg(args, "path")?;
    git(
        ctx,
        &["diff", "--no-ext-diff", "--no-textconv", &range, "--", path],
    )
}

fn show_file(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let object = format!("{}:{}", rev(args, "ref")?, str_arg(args, "path")?);
    git(ctx, &["show", &object])
}

fn log(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let n = whole_arg(args, "n").unwrap_or(1).min(i32::MAX as u64);
    let branch = rev(args, "branch")?;
    let out = git(
        ctx,
        &["log", &format!("-n{n}"), "--format=%h %as %s", branch, "--"],
    )?;
    Ok(out.trim_end().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endless_output_is_read_only_to_the_cap() {
        let mut out = Vec::new();
        read_capped(io::repeat(b'a'), &mut out).unwrap();
        assert_eq!(out.len() as u64, MAX_READ + 1);
    }
}
