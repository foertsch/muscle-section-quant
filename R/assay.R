# What R/figures.R and R/export.R share: the command line and the assay's spec. An assay is a folder R/<assay>/ with
#   measures.R  its readouts and the section values of each (what the tests load)
#   spec.R      SPEC, what the drivers need: anova_gate (optional, default TRUE: Dunnett's test only after a significant
#               one-way ANOVA; FALSE runs it directly), sizes (figure sizes in cm), tables(dir) (the tables of one run, with at
#               least palette and sections), panels(tables) (one list per figure pair: key, a named list that names it,
#               e.g. muscle and measure; measure; sv, the section values with a section number; optionally title and
#               notes), and optionally extra_figures(tables, out_dir) and extra_values(tables, write).

read_table <- function(dir, name, col_types = readr::cols(animal = "c")) {
  readr::read_csv(file.path(dir, name), show_col_types = FALSE, col_types = col_types)
}

# The assay and the folders named on the command line, with its spec and tables loaded; the output folder is made.
open_run <- function(here, script, args = commandArgs(trailingOnly = TRUE)) {
  assays <- setdiff(basename(list.dirs(here, recursive = FALSE)), "tests")
  if (length(args) != 3 || !args[1] %in% assays) {
    stop("usage: Rscript R/", script, " {", paste(assays, collapse = "|"), "} IN_DIR OUT_DIR")
  }
  source(file.path(here, "plot_functions.R"))
  for (f in c("measures.R", "spec.R")) source(file.path(here, args[1], f))
  dir.create(args[3], showWarnings = FALSE, recursive = TRUE)
  list(spec = SPEC, tables = SPEC$tables(args[2]), out_dir = args[3])
}

panel_name <- function(panel) paste(unlist(panel$key), collapse = "_")
