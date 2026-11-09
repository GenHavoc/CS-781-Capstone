# Final Report and Code Archive

The report uses the official LaTeX course template and is submitted as a PDF. It needs the repository URL, methods, experiments, results, limitations, references and the mandatory LLM assistance declaration.

The final archive name is:

`CS781-CP-[Firstname]-[Lastname]-[RollNo]-report_code.zip`

Put `code/` and `report/` directly at the ZIP root. Each phase's best-model code belongs under `code/phase1/`, `code/phase2/`, and so on. The PDF belongs at `report/CS781-CP-[Firstname]-[Lastname]-[RollNo]-report.pdf`. A GitHub source download alone is not the final submission archive.

The schedule in the course brief gives **8 November 2026** for report/code submission on Hello IITK. The active LaTeX template specifies a 10-page maximum for the main report, excluding references and the LLM Usage Declaration. An older commented version and the FAQ mention eight pages; check course announcements if the limit changes.

The [requirements checklist](../docs/requirements.md) covers the deliverables and conflicting instructions in the brief.

## Data analysis draft

[Data analysis and preprocessing](data_analysis_preprocessing.tex) contains the section supplied for the report, with LaTeX heading and reference fixes. Place it under the existing section heading in the template. Add the citations from [references.bib](references.bib) to the final bibliography. Values described as unavailable or unverified still need the matching run audit.

## Competition results inventory

The [results table](../results/README.md) records the seven scores and ranks. Finished submissions visible by phase are 14, 10, 12, 8, 8, 6 and 7, for 65 documented finished uploads. Early testing runs are included when they finished. Three Phase 7 rows are missing from the screenshot, so its final count and the total valid-submission number still need checking before the report table is finalized.

## Notebook experiment records

The [Phase 3–7 notebook run records](../experiments/notebook_run_records.md) contain transcribed local validation scores and saved run settings. They do not establish the exact Codabench submission ID for each run; resolve those joins from the corresponding checkpoint, ZIP, or job log before using them as submission-level comparisons in the report.
