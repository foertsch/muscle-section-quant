# PSR measures and the size-distribution figures. The primary readout is collagen outside the fiber outlines: red
# area in the main section outside every Cellpose-SAM outline, as a share of red + yellow area
# (collagen_outside_fraction). It depends on the outlines: red inside any outline does not count. Reads the tables
# msq.psr.summary writes: fibers.csv.gz, animals.csv, sections.csv, palette.csv.

ARTIFACTS <- 0.3  # a lab choice: sections with more fiber area in freezing-artifact tiles than this are marked in the
                  # plots (the rims of the artifact holes read as red). The mark excludes no section

# Section-level measures shown as bars. An animal's value is the mean of its sections.
BAR_MEASURES <- list(
  collagen_outside = list(value = function(s) 100 * s$collagen_outside_fraction,
                          label = "Collagen outside the fiber outlines (%)",
                          note = paste("Primary readout: red area outside every fiber outline / (red + yellow) area of the main section;",
                                       "red inside any outline does not count.")),
  fiber_count = list(value = function(s) s$n_fibers, label = "Fibers per section (exploratory)",
                     note = "Exploratory. Fibers counted in the main section (outline rule as in the size plots)."),
  fiber_density = list(value = function(s) s$n_fibers / s$section_area_mm2, label = "Fibers per mm\u00b2 (exploratory)",
                       note = "Exploratory. Fibers per mm\u00b2 of the main section.")
)

SIZE_MEASURES <- list(
  min_feret_um = list(label = "Min Feret diameter (\u00b5m)", top = 100, bw = 2),
  max_feret_um = list(label = "Max Feret diameter (\u00b5m)", top = 180, bw = 3),
  area_um2 = list(label = "Cross-sectional area (\u00b5m\u00b2)", top = 8000, bw = 150)
)

# One value per section (y), with its freezing-artifact flag when the run has a hole survey (summary --holes);
# without one, sections.csv has no freezing_artifact_area and nothing is marked.
psr_section_values <- function(sections, palette, measure) {
  sv <- section_values(sections, palette, BAR_MEASURES[[measure]])
  if (!"freezing_artifact_area" %in% names(sections)) return(sv)
  sv |> mutate(artifacts = !is.na(freezing_artifact_area) & freezing_artifact_area > ARTIFACTS)
}

# Per animal: a Gaussian kernel density with a fixed bandwidth; per genotype: mean of its animals' curves +- SEM,
# so every animal counts once however many fibers it has.
size_densities <- function(fibers, palette, column) {
  m <- SIZE_MEASURES[[column]]
  grid_n <- 512
  per_animal <- fibers |>
    filter(group %in% palette$group) |>
    group_by(group, animal) |>
    group_modify(function(d, key) {
      k <- density(d[[column]], bw = m$bw, from = 0, to = m$top, n = grid_n)
      tibble(x = k$x, y = k$y)
    }) |>
    ungroup()
  per_animal |>
    group_by(group, x) |>
    summarise(mean = mean(y), sem = sd(y) / sqrt(n()), n = n(), .groups = "drop") |>
    with_labels(palette)
}

plot_size_densities <- function(fibers, palette) {
  curves <- bind_rows(lapply(names(SIZE_MEASURES), function(col) {
    size_densities(fibers, palette, col) |> mutate(measure = SIZE_MEASURES[[col]]$label)
  })) |>
    mutate(measure = factor(measure, levels = vapply(SIZE_MEASURES, `[[`, "", "label")))
  n_animals <- fibers |> distinct(group, animal) |> count(group) |> with_labels(palette)
  legend_labels <- setNames(paste0(n_animals$label, " (n = ", n_animals$n, ")"), n_animals$label)
  ggplot(curves, aes(x, mean, colour = label, fill = label)) +
    geom_ribbon(aes(ymin = pmax(mean - sem, 0), ymax = mean + sem), colour = NA, alpha = 0.25) +
    geom_line(linewidth = 0.7) +
    scale_colour_manual(values = setNames(palette$colour, palette$label), labels = legend_labels) +
    scale_fill_manual(values = setNames(palette$colour, palette$label), labels = legend_labels) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.02))) +
    facet_wrap(~measure, scales = "free", nrow = 1, strip.position = "bottom") +
    labs(x = NULL, y = "Density",
         caption = paste0("Line: mean of the animals' distributions; band: \u00b1 SEM across animals. Smoothed with a ",
                          "Gaussian kernel of 2 \u00b5m (min Feret), 3 \u00b5m (max Feret) and 150 \u00b5m\u00b2 (area).")) +
    theme_paper() +
    theme(strip.placement = "outside", strip.background = element_blank(), strip.text = element_text(size = 9),
          axis.text.y = element_blank(), axis.ticks.y = element_blank(), legend.position = "top",
          panel.spacing = unit(0.8, "cm"))
}

# The size-distribution curves: x, then mean and SEM of the animals' densities per group.
density_table <- function(fibers, palette, column) {
  size_densities(fibers, palette, column) |>
    select(label, x, mean, sem) |>
    arrange(label) |>
    tidyr::pivot_wider(names_from = label, values_from = c(mean, sem), names_vary = "slowest",
                       names_glue = "{label} {.value}")
}
