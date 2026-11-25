# The tables of one psr run, as the Python step writes them, built from the generic palette. Nothing is random, so the
# exported values can be pinned. Another assay's run writer lives in its own helper_run_<assay>.R.
animal_table <- function(per_group = 3) {
  tibble::tibble(group = rep(palette$group, each = per_group), animal = as.character(seq_len(3 * per_group)),
                 level = rep(c(1, 2.2, 0.6), each = per_group) + rep(c(-0.2, 0, 0.3), 3))
}

write_palette <- function(dir) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(palette |> dplyr::mutate(label = as.character(label)), file.path(dir, "palette.csv"))
}

write_psr_run <- function(dir) {
  write_palette(dir)
  a <- animal_table()
  s <- a[rep(seq_len(nrow(a)), each = 2), ] |> dplyr::mutate(section = rep(1:2, nrow(a)), sample = paste0("S", dplyr::row_number()))
  s <- s |> dplyr::mutate(red_fraction = 0.05 * level + 0.01 * section, collagen_outside_fraction = 0.03 * level + 0.005 * section, n_fibers = round(900 * level) + 40 * section,
                          section_area_mm2 = 12 + 3 * section, freezing_artifact_area = ifelse(section == 1, 0.45 * (animal == "4"), NA)) |>
    dplyr::select(-level)
  readr::write_csv(s, file.path(dir, "sections.csv"))
  fibers <- a[rep(seq_len(nrow(a)), each = 40), ] |>
    dplyr::mutate(min_feret_um = 20 + 10 * level + 4 * stats::qnorm(stats::ppoints(40)), max_feret_um = 1.4 * min_feret_um,
                  area_um2 = pi * (min_feret_um / 2)^2) |>
    dplyr::select(-level)
  readr::write_csv(fibers, file.path(dir, "fibers.csv.gz"))
}

# md5 of every CSV a driver wrote, after rounding the doubles to 8 decimals (the smoothed curves carry noise below that).
# To pin again after an intended change: write these into export_pins.csv (export_pins_<assay>.csv for another assay).
export_pins <- function(dir) {
  files <- sort(list.files(dir, pattern = "\\.csv$"))
  vapply(files, function(f) {
    d <- readr::read_csv(file.path(dir, f), show_col_types = FALSE) |>
      dplyr::mutate(dplyr::across(dplyr::where(is.double), \(v) round(v, 8) + 0))
    tmp <- tempfile(fileext = ".csv")
    on.exit(unlink(tmp))
    readr::write_csv(d, tmp, na = "")
    unname(tools::md5sum(tmp))
  }, "")
}
