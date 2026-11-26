# The desktop app

The app puts chat, documents, tasks and your models in one window. It runs the same engine as the
`larch` command, in the same process, so nothing runs in the background when the app is closed. It
is a preview.

![The Home screen, with the app's sample data](assets/app-home-light.png#gh-light-mode-only)
![The Home screen, with the app's sample data](assets/app-home-dark.png#gh-dark-mode-only)

## The sidebar

| Item | What's there |
|---|---|
| **Home** | Start a chat, see what needs your approval, pick up the last chat, and check GPU memory |
| **Chats** | Every saved chat. Search, rename, pin or archive them |
| **Tasks** | Jobs that run on their own, on a schedule or when something happens. Describe one in words and Larch drafts the steps |
| **Models** | The models you have, new ones to download, and what fits on this computer |
| **Tools** | Folders the agent may use, web search keys and MCP servers |
| **Settings** | Appearance, memory, the assistant's behaviour and engine options |

The status card at the bottom of the sidebar shows the loaded model and its speed. The small tree in
the corner is the assistant: it turns gold from the bottom while it works and its top tier turns
ember when it needs you. Every state is also written in words beside it.

## Chats

Type in the box at the bottom. The buttons under it add context to the next message:

| Button | Effect |
|---|---|
| **+ Attach** | Add files or a folder. See [Documents](documents.md) |
| **Research** | Search the web, read several sources and write a report with citations |
| **Code** | Work in a code repository, with the sandbox from [The agent](agent.md#code-mode-and-the-sandbox) |
| **Web** | Let the model search the web for this message |
| **Folder** | Give the chat a folder to read and write |

When the agent wants to write a file or run a command, the chat pauses and the request appears in
**Needs you** on Home and in the chat.

## Models

**Library** lists the models on disk. **Discover** searches Hugging Face; each result shows whether it
fits your GPU before you download it. **This computer** shows the GPU, its memory, and how the loaded
model uses it: weights, chats and free space.

On a computer with no model yet, Home offers up to three that suit its GPU, each with a **Download**
button, and **Add a folder** for GGUF files you already have. The first model you download loads by
itself.

Switching models unloads the current one first. Chats keep their history but start a fresh context.

### Faster on this computer

Some models are larger than the GPU. For those Larch knows how to speed up, the model's card offers
**Faster on this computer** on an NVIDIA GPU that cannot hold the whole file. **Make it faster** writes
a second copy of the model in which the experts it rarely uses are stored in less detail, so all of
it fits on the GPU. Quality is slightly lower; the copy takes a few minutes and about as much disk as
the model. Chats keep running on the original meanwhile, and the app switches to the copy when it is
ready. **Use the original** switches back. The original file is never changed.

![The faster version on a model card: offered, being made, and in use](assets/app-faster-light.png#gh-light-mode-only)
![The faster version on a model card: offered, being made, and in use](assets/app-faster-dark.png#gh-dark-mode-only)

After switching, the app checks that the copy loaded entirely on the GPU. If another program holds
too much GPU memory for that, the copy would run slower than the original, so the app loads the
original again and keeps the copy for later. Deleting a model deletes its faster copy too.

## Tasks

A task is a list of steps that runs without you: read a folder, ask the model, decide, notify you.
Build one in **Tasks**, use **Test run** to try it, then **Turn on**. A task that needs approval
stops and waits in **Needs you**.

## Keyboard

On macOS use `Cmd` where this table says `Ctrl`.

| Keys | Action |
|---|---|
| `Ctrl N` | New chat |
| `Ctrl L` | Go to the message box |
| `Enter` / `Shift Enter` | Send / new line |
| `Ctrl .` | Stop the reply |
| `Ctrl 1`, `Ctrl 2`, `Ctrl 3` | Home, Chats, Models |
| `Ctrl ,` | Settings |
| `Esc` | Close a popover or dialog |
| `Ctrl Q` | Quit |

**Settings** lists the same shortcuts.

## Where the app keeps things

Chats, settings, memory and downloaded models are in your data directory, listed in
[Configuration](configuration.md#where-data-is-kept). Uninstalling keeps them unless you pass
`--purge`.
