//! `cuprum` prints its version, or with `--schema` the protocol's JSON Schema.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        [] | ["--version"] => println!("cuprum {}", env!("CARGO_PKG_VERSION")),
        ["--schema"] => println!("{:#}", cuprum::proto::schema()),
        _ => {
            eprintln!("usage: cuprum [--version | --schema]");
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}
