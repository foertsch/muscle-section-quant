# The two drivers on a small psr run (helper_run.R). The exported values are pinned to what the per-assay script wrote
# before the drivers were merged (export_pins.csv); every figure is written, in both formats.
library(testthat)
source("helper_palette.R")
source("helper_run.R")
source("helper_drive.R")

test_that("export.R writes the values the psr script wrote", {
  expect_pinned_export("psr", "export_pins.csv")
})

test_that("figures.R writes a pair of figures per panel, the statistics and what only psr draws", {
  psr <- drive("figures.R", "psr")
  expect_setequal(list.files(psr), c(figure_files(c("collagen_outside", "fiber_count", "fiber_density")),
                                     "fiber_size_density.pdf", "fiber_size_density.png", "stats_vs_control.csv"))
  expect_equal(names(readr::read_csv(file.path(psr, "stats_vs_control.csv"), show_col_types = FALSE))[1:3],
               c("measure", "group", "test"))
})

test_that("an unknown assay is refused with the list of the known ones", {
  assays <- setdiff(basename(list.dirs("..", recursive = FALSE)), "tests")  # as assay.R lists them
  out <- suppressWarnings(system2(file.path(R.home("bin"), "Rscript"), c("../export.R", "nope", "a", "b"), stdout = TRUE, stderr = TRUE))
  known <- paste0("\\{", paste(assays, collapse = "\\|"), "\\}")
  expect_match(paste(out, collapse = "\n"), paste0("usage: Rscript R/export.R ", known, " IN_DIR OUT_DIR"))
})
