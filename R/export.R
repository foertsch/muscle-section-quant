# The numbers behind every figure of one run, as CSV, to replot them elsewhere (e.g. Prism or Excel).
#
# Usage:
#   Rscript R/export.R ASSAY IN_DIR OUT_DIR
# Same arguments and panels as figures.R, so the values are exactly the plotted ones. Per panel (psr: one per measure):
# <name>_sections.csv (points of the by-animal plot), <name>_animals.csv (bars of the by-animal plot: mean of sections
# +- SEM; also the points of the by-genotype plot), <name>_animals_by_group.csv (the same animal values, one column per
# group, as a Prism column table), <name>_groups.csv (bars of the by-genotype plot: mean +- SEM). Plus
# stats_vs_control.csv with group labels, and what the assay's spec adds (psr: fiber_size_density_<measure>.csv, the
# smoothed curves: mean and SEM per group).

here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))
source(file.path(here, "assay.R"))
run <- open_run(here, "export.R")
palette <- run$tables$palette
write <- function(df, name) write_values(df, run$out_dir, name)
flag <- function() across(any_of("artifacts"), .names = "freezing_artifacts")  # psr marks animals with freezing artifacts

stats <- list()
for (p in run$spec$panels(run$tables)) {
  name <- panel_name(p)
  unit <- p$measure$label
  av <- animal_values(p$sv)
  write(p$sv |> arrange(label, animal, section) |> transmute(group = label, animal, section, !!unit := y, flag()),
        paste0(name, "_sections"))
  write(av |> arrange(label, animal) |> transmute(group = label, animal, n_sections, !!unit := y, sem_of_sections = sem, flag()),
        paste0(name, "_animals"))
  write(wide_by_group(av, palette), paste0(name, "_animals_by_group"))
  write(group_summary(av) |> rename(group = label), paste0(name, "_groups"))
  stats[[name]] <- dunnett_vs_control(av, palette, !isFALSE(run$spec$anova_gate)) |>
    mutate(!!!p$key, .before = 1) |>
    mutate(group = as.character(palette$label[match(group, palette$group)]),
           control = as.character(palette$label[palette$control]), .after = group)
}
write(bind_rows(stats), "stats_vs_control")
if (!is.null(run$spec$extra_values)) run$spec$extra_values(run$tables, write)
cat("wrote values to", run$out_dir, "\n")
