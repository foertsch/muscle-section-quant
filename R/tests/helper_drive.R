# Run one driver (figures.R or export.R) on a small run of one assay (write_<assay>_run, helper_run.R) and return its
# output folder.
drive <- function(script, assay) {
  in_dir <- tempfile()
  out_dir <- tempfile()
  match.fun(paste0("write_", assay, "_run"))(in_dir)
  out <- suppressWarnings(system2(file.path(R.home("bin"), "Rscript"), c(file.path("..", script), assay, in_dir, out_dir),
                                  stdout = TRUE, stderr = TRUE))
  expect_null(attr(out, "status"), info = paste(out, collapse = "\n"))
  out_dir
}

figure_files <- function(names) {
  as.vector(outer(names, c("_by_genotype.pdf", "_by_genotype.png", "_by_animal.pdf", "_by_animal.png"), paste0))
}

# The exported values of one assay against its pins (export_pins.csv for psr, export_pins_<assay>.csv for another).
# Both sides are put in C-locale order (radix), since sort() follows the locale and Linux orders "_" and "." unlike macOS.
expect_pinned_export <- function(assay, pins_file) {
  want <- readr::read_csv(pins_file, show_col_types = FALSE)
  want <- want[want$assay == assay, ]
  got <- export_pins(drive("export.R", assay))
  expect_equal(got[order(names(got), method = "radix")],
               setNames(want$md5, want$file)[order(want$file, method = "radix")], info = assay)
}
