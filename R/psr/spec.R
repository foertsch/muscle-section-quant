# What the figure and export drivers read from one PSR run (the tables msq.psr.summary writes) and draw from it.

SPEC <- list(
  anova_gate = FALSE,  # Dunnett's test against the control directly on the animal means, no ANOVA first
  sizes = list(genotype = c(7.5, 9), animal = c(15, 8)),
  tables = function(dir) {
    list(palette = read_palette(file.path(dir, "palette.csv")), sections = read_table(dir, "sections.csv"),
         fibers = read_table(dir, "fibers.csv.gz"))
  },
  panels = function(t) {
    lapply(names(BAR_MEASURES), function(m) {
      list(key = list(measure = m), measure = BAR_MEASURES[[m]], sv = psr_section_values(t$sections, t$palette, m))
    })
  },
  extra_figures = function(t, out_dir) {
    save_figure(plot_size_densities(t$fibers, t$palette), file.path(out_dir, "fiber_size_density"), 18, 7)
  },
  extra_values = function(t, write) {
    for (column in names(SIZE_MEASURES)) {
      write(density_table(t$fibers, t$palette, column) |> rename(!!SIZE_MEASURES[[column]]$label := x),
            paste0("fiber_size_density_", column))
    }
  }
)
