# Paper figures from the tables of one run.
#
# Usage:
#   Rscript R/figures.R ASSAY IN_DIR OUT_DIR
# ASSAY is one of the folders of R/ (each with measures.R and spec.R). IN_DIR holds the tables the assay's Python step
# wrote: for psr fibers.csv.gz, sections.csv and palette.csv (msq.psr.summary). Writes to OUT_DIR, each as .pdf and
# .png, <name>_by_genotype and <name>_by_animal for every panel of R/<assay>/spec.R (psr: collagen_outside,
# fiber_count, fiber_density) and, for psr, fiber_size_density; plus stats_vs_control.csv (one-way ANOVA, Dunnett's test against the control
# group in palette.csv, and the smallest detectable difference; animals as the unit). Dunnett's test runs only if the
# ANOVA is significant, unless the assay's SPEC sets anova_gate = FALSE (psr): then it runs directly and the table has
# a test column saying so.

here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))
source(file.path(here, "assay.R"))
run <- open_run(here, "figures.R")
palette <- run$tables$palette

stats <- list()
for (p in run$spec$panels(run$tables)) {
  av <- animal_values(p$sv)
  st <- dunnett_vs_control(av, palette, !isFALSE(run$spec$anova_gate))
  stats[[panel_name(p)]] <- st |> mutate(!!!p$key, .before = 1)
  artifacts <- "artifacts" %in% names(p$sv)
  base <- file.path(run$out_dir, panel_name(p))
  size <- run$spec$sizes
  save_figure(plot_by_genotype(av, palette, st, p$measure, p$title, p$notes, artifacts),
              paste0(base, "_by_genotype"), size$genotype[1], size$genotype[2])
  save_figure(plot_by_animal(p$sv, palette, p$measure, p$title, p$notes, artifacts),
              paste0(base, "_by_animal"), size$animal[1], size$animal[2])
}
stats <- bind_rows(stats)
readr::write_csv(stats, file.path(run$out_dir, "stats_vs_control.csv"))
print(stats, width = Inf)
if (!is.null(run$spec$extra_figures)) run$spec$extra_figures(run$tables, run$out_dir)
cat("wrote figures to", run$out_dir, "\n")
