# 53 — Background tasks: a long answer is asked for, then fetched

**Status:** planned 2026-10-06 — big picture, to groom before building
**Depends on:** 52 (the reading of a picture, its first use)

Some things drift is asked take minutes and end in an answer, not in a
picture: the assistant's reading of a picture for its redraw (52) took 98 s
on an 8192² picture, and a 16384² one is four and a half times the tiles.
Asked as one
request that stays open until the answer is in, it is at the mercy of every
wait on the way — the browser's client gave up after one minute, the server
after five, each time as "the operation was aborted". Waiting longer
everywhere (`RedrawPlan.ReadingMinutes`) is a patch: a reload loses the
reading, nothing can stop it, and the next long thing will meet the same
walls.

So: **a long answer is a task.** Asking for it returns at once; the task runs
on the backend; whoever asked fetches what became of it.

## What it does

- **Asking returns at once**, with the task: its id, what it is, what it is
  about, and that it is running.
- **The task runs on the backend**, on its own fork (ox, as the jobs do), until
  it has its answer or the reason it has none.
- **Its state is fetched**, not waited for: running — with a word on what it is
  doing and, where the task can tell, how far along it is — done with its
  answer, failed with its reason, or stopped.
- **It can be stopped.** What stopping means is the task's own business (the
  reading drops its call to the assistant).
- **It outlives the page.** A task belongs to what it is about — a reading to
  its picture and its tiles — so the page that comes back to that picture finds
  the reading running, or done, and does not ask again.
- **Nothing is kept for ever.** A task that ended is forgotten some time after
  its answer was last fetched, and drift restarted forgets them all: a task is
  cheap to ask again, and an answer worth keeping is kept by whoever asked for
  it, where it belongs.

## For the reading (52), nothing changes on screen

The redraw panel is as it is: **🤖 Read the picture**, the button spinning
while the assistant reads, then the line that sums the reading up and the
tiles. No job list, no new control. Underneath, the button starts the task and
the panel fetches its state every two seconds; opening the picture again while
its reading runs shows the button spinning, and a reading already made for
these tiles is there without asking.

## Shape

- **One system, whatever the task.** `Tasks` (backend) knows nothing of
  readings: `start(kind, subject)(work)` records a task, runs `work` on a fork
  and keeps what it returns — an answer, encoded as JSON by the caller's own
  codec, or a reason. `work` is handed what it needs to report (`doing`,
  `progress`) and to learn it was stopped. A kind is a word (`"redraw-plan"`);
  a subject is whatever tells two askings of the same thing apart (the
  picture, its tile size and grid offset), so the same asking finds its task
  instead of starting a second.
- **`shared`:** `Task(id, kind, subject, state, doing, progress, startedAt,
  endedAt, error)` with `TaskState` = running, done, failed, stopped; and the
  endpoints, none of which names a kind:
  - `GET /api/tasks/{id}` — the task;
  - `GET /api/tasks?kind=&subject=` — the tasks of a kind about a subject,
    newest first: how a page finds its own;
  - `POST /api/tasks/{id}/stop`.
- **Each kind keeps its own two endpoints**, typed as today, so that a task's
  answer is never a blob the browser has to guess the shape of:
  - the one that starts it — `POST /api/outputs/{date}/{file}/redraw-plan`
    answers the `Task` instead of the `RedrawPlan`;
  - the one that gives its answer — `GET /api/tasks/{id}/redraw-plan` answers
    the `RedrawPlan` of a done task, and why not otherwise.
- **Frontend:** `TaskService` — `await(task): EventStream[Task]`, which polls a
  task until it ends, and `find(kind, subject)`. `AutoRedrawCard` starts, finds
  and awaits through it and keeps its button and its lines.
- The request is never open for more than a moment: `RedrawPlan
  .ReadingMinutes` and the three waits it lengthened (the browser's request,
  the server's request and idle timeouts) go back to what they were; the call
  to the assistant keeps a long timeout of its own, which is the task's.

## Not this

- **Post-process jobs** (15, 40) stay what they are: a job makes a picture, is
  listed in the gallery, can be paused and resumed and is written to disk. A
  task makes an answer and none of that. If a job ever needs a long answer on
  the way (a redraw that reads its picture first), it asks a task.
- **Sessions and downloads** have their own lists and lives.

## To settle when grooming

- **What else becomes a task.** Candidates today, each a request that stays
  open: the inspection of a cached file before a conversion (25), a picture's
  import when it is large (bug 41 is its size limit, not its time), the
  assistant writing a prompt (32). None is known to time out; they move when
  they do, or when moving them is free.
- **Progress for the reading.** The assistant's answer is one JSON object;
  streamed, the backend could count the tiles already answered. Worth it at
  289 tiles; a second step.
- **Stop on screen.** The system has it; the reading's panel does not show it
  ("keep the ergonomics as it is"). A reading of ten minutes or more may want one.
- **How long an ended task is remembered**, and whether a done reading should
  rather be kept with the picture (it is its tiles' settings) than as a task.
