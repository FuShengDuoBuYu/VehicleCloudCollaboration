---
name: vehicle-cloud-thesis
description: Work on Guo Zhongtian's vehicle-cloud-collaboration professional-master thesis, including project research, evidence audits, literature review, experiment analysis, code changes, and LaTeX drafting in the thesis workspace. Use whenever a task concerns PROJECT_INDEX.md, VehicleCloudCollaboration, papers, the Raspberry Pi car, the cloud API, or the thesis manuscript.
---

# Vehicle–Cloud Thesis

Treat this as a long-running research project whose code, experiments, sources, and prose must remain mutually traceable.

Use the thesis root's `AGENTS.md` for chat routing, proactive reminders, shared-file ownership and persistence conventions. For work crossing chats or hosts, use `skills/thesis-handoff/SKILL.md`; save only task-relevant context and do not assume the remote project shares the local transcript or files. If working from a car-only checkout, use the actual installed car-side guidance and supplied handoff materials instead of assuming these local-parent paths exist.

## Start from durable context

Find the nearest project root containing `PROJECT_INDEX.md` and read the task-relevant portions before substantive research, writing, code or experiment work. Use `papers/MATERIAL_INDEX.md` when locating or interpreting opening, midterm, administrative, or local-paper materials. Use `EXPERIMENT_VALIDATION_PLAN.md` as the durable backlog and evidence template for implementation or validation work. Read only the relevant code, experiment records, and `papers/` sources after that; a trivial edit does not require a full-project audit. If current evidence changes a durable fact, update the corresponding index or plan in the same task when appropriate.

Use these confirmed project facts unless the user revises them:

- The opening report and midterm report use the same title: `基于云–端协同的大模型增强智能驾驶长尾场景感知与决策方法研究`. Treat this as the confirmed thesis title unless the user or school later changes it.
- The target system combines the current on-vehicle LCC + YOLOPv2 pipeline, a new temporal trigger based on abrupt changes in fresh YOLOPv2 drivable-area masks, and cloud recognition exposed through `https://api.fushengduobuyu.com/docs`.
- No preserved validation result currently supports the temporal long-tail trigger. Treat it as a proposed operational proxy requiring video-level pre-calibration followed by repeated live-run validation, not as a completed detector or a universal semantic definition of long-tail scenes.
- The confirmed target sequence is: abrupt mask-change candidate → non-blocking gradual deceleration or immediate stop according to locally calibrated severity → asynchronous structured cloud scene/risk/advice response → local validation and arbitration → stable-perception recovery gate → local autonomous driving. Timeout, invalid/old responses, local hazards, and watchdog conditions keep or force the vehicle stopped.
- Keep model-generated scene/risk/advice separate from the locally generated request, timing, and arbitration envelope. Cloud output never directly sets PWM. Adjacent-corridor planning, limited left/right avoidance, and lane-change execution are planned supplemental experiments rather than permanently excluded functions. Until they are implemented, calibrated, and independently verified through replay and wheels-up safety gates followed by low-speed closed-course live runs, left/right cloud advice is record-only and the vehicle remains stopped; after validation, local planning and safety gates still decide execution.
- State that traffic-light or traffic-director conflicts, multi-vehicle interaction, and other broad semantic long-tail cases are outside the primary experiments at this stage because the current closed course lacks the required traffic facilities and controllable participants. Present this as an environment-limited experimental boundary, not an intrinsic or permanent exclusion.
- Describe current HEAD, historical validation, and the desired integrated system separately until code and end-to-end experiments prove they are one running chain.
- The user manually annotated Long-tail/Non-long-tail. The working heuristic combines scene visibility with whether the content is absent or underrepresented in conventional training data. Present this as a human annotation criterion, not an objective universal definition.
- Continue to compare research scope and claims against both the opening report and midterm report in `papers/`; record architecture evolution or conflicts instead of silently merging versions.
- Preserve the thesis template's original `style/font=times` typography, which this legacy fduthesis version implements with XITS text and math fonts. Do not silently substitute Latin Modern or another font to make a build pass. If XITS is unavailable, identify and install the dependency, keep any fallback PDF labeled as a temporary environment smoke test, and revalidate the original font before treating the manuscript build as accepted. This preference does not override a later explicit school formatting requirement.

## Experimental execution

- Treat desktop replay, Raspberry Pi motor-disabled replay, fault injection, and wheels-up tests as development, pre-calibration, and pre-motion safety gates. They may support secondary controlled analyses, but never describe them as the primary or final evidence for the target vehicle--cloud method.
- The final primary evidence for temporal triggering during operation, cloud-assisted scene handling, local arbitration, deceleration/stopping/recovery, complete-system ablations, and limited avoidance must come from repeated live runs of the physical car at the minimum safe speed on the controlled closed course. The target algorithm must run online on the vehicle; preserve synchronized onboard logs, onboard and external video, run-level ground truth, configurations, failures, and outcomes.
- Use real-car recordings for debugging and trigger pre-calibration, then freeze the intended formal-test code, model, calibration, thresholds, and protocol before the live runs. Recordings and logs from those formal live runs may be analyzed offline for event-level metrics. Do not tune on final-test runs or count adjacent frames as independent samples.
- Keep capture run IDs, replay/precheck run IDs, and formal live-run IDs linked in a manifest. Preserve raw video, original logs, replay logs, dropped-frame behavior, software/model/config hashes, and manual event annotations. Split and resample by independent run or event, not by adjacent frames.
- Controlled counterfactual trigger tests may modify copies of real car recordings to isolate one visual factor. Never overwrite the original recording or log. Record the source hash, transformation, parameters, mask or region, random seed when applicable, and derived-file hash; label the result as synthetic/controlled-derived data and report it separately from naturally recorded events. Such tests can support sensitivity and fault-isolation claims, but cannot establish real-scene prevalence, physical stopping, collision avoidance, or full-system safety.
- Video replay can debug perception, triggering, cloud requests, arbitration transitions, and virtual PWM, and can provide secondary paired comparisons. It cannot establish the final operating performance of the integrated system or prove physical stopping distance, wheel dead-zone behavior, collision avoidance, or real recovery motion.
- Exercise exhaustive or hazardous communication/model fault cases first without actuation. Repeat only a safely controllable subset during minimum-speed closed-course runs to verify real stopping and recovery; never create an unsafe moving-car condition merely to make the fault matrix look complete.

## Evidence and writing integrity

Academic polishing is allowed: improve structure, terminology, transitions, explanations, tables, and defensible interpretation of existing data. It is also acceptable to calculate new descriptive statistics from preserved raw data and clearly identify reasoned inferences.

Never invent or silently fill in experiments, measurements, sample counts, success rates, baselines, citations, quotations, implementation status, or regulatory requirements. Do not present a planned integration, historical demo, mock result, or screenshot-only claim as a current verified result. When evidence is incomplete, state the limitation and what would verify it.

For every important quantitative claim, preserve or establish a route to the code commit, configuration, input/run ID, raw log, and analysis method. Use the `E1–E4/X` evidence vocabulary defined in `PROJECT_INDEX.md` when it helps distinguish claim strength.

## Manuscript presentation

- Use the author-confirmed seven-chapter structure unless the author explicitly revises it: (1) 绪论; (2) 相关工作; (3) 车云协同系统总体设计; (4) 可行驶区域时序触发与本地安全响应; (5) 云端多模态场景理解与车端安全仲裁方法; (6) 系统实现与实验评估; and (7) 总结与展望. Keep Chapter 3 at architecture level, Chapter 4 focused on local triggering and immediate safety response, Chapter 5 focused on cloud understanding and vehicle-side arbitration, and Chapter 6 focused on implementation evidence and experimental results. Do not create empty results merely to balance the chapters.
- Write the thesis manuscript as a scholarly document, not as a project-status log. Do not leave words such as “初稿”, “占位”, “待定”, or “待验证”, empty result tables, editing instructions, or completion checklists in the reader-facing `.tex` files. Keep unconfirmed administrative fields, missing experiments, evidence gaps, and follow-up actions in the root `THESIS_COMPLETION_NOTES.md` and the experiment backlog instead. This separation never permits an unsupported result to be stated as completed: omit the claim, describe only the method or evaluation protocol, or explicitly delimit what the available evidence establishes without using editorial placeholders.
- Prefer plain, formal Chinese. Keep one main idea per sentence, translate avoidable jargon, and explain necessary technical terms where they first matter. Preserve technical precision without making the prose needlessly abstract.
- Retain only formulas that define the method, controller, trigger, arbitration, or evaluation metric. Define every symbol immediately after the formula and follow it with a plain-language interpretation; use a small example when that materially improves comprehension. Do not add mathematical complexity merely to make the manuscript appear more technical.
- Treat the Chinese abstract, English abstract, and main text as independently readable units. At the first occurrence of an English abbreviation in the Chinese abstract and again in the main text, use `中文名称（English Full Name，ABBREVIATION）`; in the English abstract, write the English full name before the abbreviation. Expand uncommon abbreviations in independently read figure or table captions when needed. Thereafter use one consistent abbreviation and capitalization. Standard units and universally recognized mathematical symbols do not need artificial expansions.
- Favor same-protocol internal baselines, ablations, confidence intervals, and repeated-run statistics for causal comparisons. Published external numbers may be included only after verification against a primary source; state the dataset, task, metric, input, hardware, and protocol differences that limit direct comparison. Never imply an apples-to-apples ranking when those conditions differ.
- Use the user's real vehicle, course, recorded video, logs, and model outputs for experimental figures. Record the source run, frame or timestamp, model/configuration, and processing method in the caption or a provenance note. Never use an AI-generated, staged, or merely illustrative image as experimental evidence.

## Literature and citations

Actively search the internet when drafting or revising background, related work, technical comparisons, standards, or other citation-dependent claims. Prefer original papers and authoritative publisher, conference, DOI, arXiv, dataset, model, or standards pages. Verify title, authors, year, venue, identifier, and the exact claim supported before adding a reference.

Use the user's local papers where relevant, but do not treat a filename or secondary summary as proof of a paper's contents. Avoid citation padding: each reference must have a stated role in the argument. Keep bibliographic records deduplicated and make them traceable to the manuscript citation key.

## Operational boundaries

- Treat the Raspberry Pi, ignored outputs, model weights, and cloud service as valuable research assets. Read-only inspection is allowed when in scope; do not start motors, launch live-motion experiments, send inference requests, alter services, repair Git, or delete data without the user's authorization for that action.
- The user authorizes selective, necessary copies of irreplaceable car-side evidence to the thesis workstation. Before copying, resolve exact source/target paths, inspect size and free space, prefer manifests/logs/configs and experiment-relevant recordings, preserve timestamps, and generate source/destination checksums. Never delete or overwrite the car-side source. A full bulk copy still requires an explicit scope decision.
- Perform the pre-motion Raspberry Pi replay when car-side timing or deployment behavior matters, even when a workstation backup exists. Label both workstation and Raspberry Pi replay as preliminary/secondary evidence; neither substitutes for the formal physical-car runs.
- Viewing remote data in place is acceptable, but flag irreplaceable or corrupted evidence and explain the concrete consequence of having no independent copy.
- Never persist passwords, API keys, tokens, or private endpoint credentials in the skill, index, manuscript, commands intended for reuse, or Git.
- Preserve unrelated user edits and distinguish CRLF-only changes from substantive changes.
- Manage the thesis workspace with Git. Keep the existing vehicle-code repository as a separately traceable repository/submodule, and keep raw videos, replay outputs, backups, secrets, third-party paper PDFs, and unredacted personal/administrative source files out of ordinary Git history unless the user deliberately chooses an appropriate private/LFS storage plan.

## Durable new requirements

When the user states a clear rule intended to apply consistently across future thesis tasks, put collaboration and routing rules in the thesis root's `AGENTS.md`, repeatable research or writing procedures in this skill (or a focused supporting reference), and verified facts in the relevant index. Keep task-local requests and tentative preferences local. Validate changed skills and briefly tell the user what was persisted.
