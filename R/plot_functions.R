# Shared figure and statistics functions for the assays' R figures (R/<assay>/), in R so the panels match
# the lab's other figures. They only read the tables the Python pipelines write.
#
# A measure is list(label = <axis label>, note = <caption sentence>, value = function(sections) <one number per
# row>). Readouts are per section; an animal's value is the mean of its sections; groups compare animals: one-way
# ANOVA on animal values, then Dunnett's test of each group against the control only if the ANOVA is significant
# (not for psr, whose SPEC sets anova_gate = FALSE);
# next to it the smallest difference to the control that a two-sample t-test with these animal numbers and this
# spread would detect (80% power, alpha 0.05).

library(ggplot2)
library(dplyr)

ALPHA <- 0.05   # post-hoc tests run only when the one-way ANOVA reaches this
POWER <- 0.8    # for the smallest detectable difference

theme_paper <- function(base_size = 9) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(axis.line = element_line(linewidth = 0.4), axis.ticks = element_line(linewidth = 0.4),
          axis.text = element_text(colour = "black"), legend.title = element_blank(),
          legend.key.size = unit(0.35, "cm"),
          plot.title = element_text(size = base_size + 1, face = "bold"),
          plot.caption = element_text(size = base_size - 2.5, colour = "grey35", hjust = 0))
}

read_palette <- function(path) {
  p <- readr::read_csv(path, show_col_types = FALSE)
  p$label <- factor(p$label, levels = p$label)
  p
}

with_labels <- function(df, palette) {
  df |>
    inner_join(palette |> select(group, label), by = "group") |>
    mutate(label = factor(label, levels = levels(palette$label)))
}

stars <- function(p) {
  cut(p, breaks = c(-Inf, 1e-4, 1e-3, 1e-2, 0.05, Inf), labels = c("****", "***", "**", "*", "ns"), right = FALSE) |>
    as.character()
}

# One value (y) per section of the palette's groups; keep: a logical over the rows (e.g. one muscle).
section_values <- function(sections, palette, measure, keep = TRUE) {
  s <- sections[keep & sections$group %in% palette$group, ]
  s$y <- measure$value(s)
  with_labels(s, palette)
}

# One value per animal: the mean of its sections, with the SEM of the sections. When the sections carry an
# artifacts flag, an animal is marked when at least half of its sections are flagged.
animal_values <- function(sv) {
  av <- sv |>
    group_by(group, label, animal) |>
    summarise(sem = ifelse(n() > 1, sd(y) / sqrt(n()), 0), y = mean(y), n_sections = n(), .groups = "drop")
  if ("artifacts" %in% names(sv)) {
    flags <- sv |>
      group_by(group, label, animal) |>
      summarise(artifacts = mean(artifacts) >= 0.5, .groups = "drop")
    av <- av |> left_join(flags, by = c("group", "label", "animal"))
  }
  av
}

# Smallest difference to the control a two-sample t-test would detect, from the ANOVA's residual SD and the harmonic
# mean of the two group sizes. A guide for planning, not adjusted for the Dunnett comparisons.
detectable_difference <- function(sigma, n_group, n_control, power = POWER, alpha = ALPHA) {
  n <- 2 / (1 / n_group + 1 / n_control)
  vapply(n, function(k) stats::power.t.test(n = k, sd = sigma, sig.level = alpha, power = power)$delta, 0)
}

# One-way ANOVA on animal values; only if it is significant, Dunnett's test of every other group against the
# control. Otherwise the post-hoc columns stay NA (posthoc = FALSE) and the plots show no stars. gate = FALSE (an
# assay's choice, SPEC$anova_gate) runs Dunnett's test directly on the animal values; the ANOVA p is still reported,
# and a column test says which test ran. With the default gate = TRUE the output is as before, with no test column.
dunnett_vs_control <- function(av, palette, gate = TRUE) {
  control <- palette$group[palette$control]
  present <- intersect(palette$group, unique(av$group))
  d <- av |> mutate(group = factor(group, levels = c(control, setdiff(present, control))))
  fit <- aov(y ~ group, data = d)
  anova_p <- summary(fit)[[1]][["Pr(>F)"]][1]
  sigma <- sqrt(sum(residuals(fit)^2) / fit$df.residual)
  others <- setdiff(levels(d$group), control)
  n_animals <- as.integer(table(d$group)[others])
  n_control <- as.integer(table(d$group)[[control]])
  base <- tibble(group = others, anova_p = anova_p, n_animals = n_animals, n_control = n_control,
                 residual_sd = sigma, detectable_difference = detectable_difference(sigma, n_animals, n_control))
  if (gate && anova_p >= ALPHA) {
    return(base |> mutate(difference = NA_real_, ci_low = NA_real_, ci_high = NA_real_, p_adjusted = NA_real_,
                          stars = NA_character_, posthoc = FALSE, .after = group))
  }
  test <- multcomp::glht(fit, linfct = multcomp::mcp(group = "Dunnett"))
  dn <- summary(test)
  ci <- confint(test)$confint
  out <- tibble(group = sub(paste0(" - ", control), "", names(dn$test$coefficients)),
         difference = unname(dn$test$coefficients), ci_low = ci[, "lwr"], ci_high = ci[, "upr"],
         p_adjusted = unname(dn$test$pvalues), stars = stars(unname(dn$test$pvalues)), posthoc = TRUE) |>
    left_join(base, by = "group")
  if (gate) out else mutate(out, test = "Dunnett vs control, no ANOVA gate", .after = group)
}

test_note <- function(stats, control = "control") {
  p <- stats$anova_p[1]
  p_text <- if (p < 0.001) formatC(p, format = "e", digits = 1) else formatC(p, format = "f", digits = 3)
  if ("test" %in% names(stats)) paste0("Stars: Dunnett's test vs ", control, " on animal means, no ANOVA first.")
  else if (isTRUE(stats$posthoc[1])) paste0("One-way ANOVA p = ", p_text, "; stars: Dunnett's test vs ", control, ".")
  else paste0("One-way ANOVA p = ", p_text, " (not significant); no post-hoc test.")
}

# Each sentence on its own line(s), wrapped to fit a narrow panel.
wrap_caption <- function(sentences, width) {
  paste(unlist(lapply(sentences, strwrap, width = width)), collapse = "\n")
}

artifact_shapes <- function(what) {
  scale_shape_manual(values = c(`FALSE` = 21, `TRUE` = 4),
                     labels = c(`FALSE` = what, `TRUE` = paste(what, "with heavy freezing artifacts")), drop = FALSE)
}

# Bars: mean of the animals with SEM; points: animals. title: e.g. the muscle; notes: extra caption sentences;
# artifacts: draw flagged animals with a cross and a legend.
plot_by_genotype <- function(av, palette, stats, measure, title = NULL, notes = character(), artifacts = FALSE) {
  s <- av |>
    group_by(group, label) |>
    summarise(mean = mean(y), sem = sd(y) / sqrt(n()), top = max(c(y, mean + sem), na.rm = TRUE), n = n(),
              .groups = "drop") |>
    left_join(stats |> select(group, stars), by = "group")
  ymax <- max(s$top) * 1.18
  points <- if (artifacts) {
    geom_point(data = av, aes(y = y, shape = artifacts), fill = "white", colour = "black", size = 1.6, stroke = 0.5,
               position = position_jitter(width = 0.12, height = 0, seed = 1))
  } else {
    geom_point(data = av, aes(y = y), shape = 21, fill = "white", colour = "black", size = 1.6, stroke = 0.5,
               position = position_jitter(width = 0.12, height = 0, seed = 1))
  }
  p <- ggplot(s, aes(label, mean)) +
    geom_col(aes(fill = label), width = 0.62, colour = "black", linewidth = 0.35) +
    geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = 0.22, linewidth = 0.4) +
    points +
    geom_text(data = s |> filter(!is.na(stars)), aes(y = top + 0.04 * ymax, label = stars), size = 3.4, vjust = 0) +
    scale_fill_manual(values = setNames(palette$colour, palette$label), guide = "none") +
    scale_y_continuous(limits = c(0, ymax), expand = expansion(mult = c(0, 0))) +
    scale_x_discrete(labels = function(x) paste0(x, "\n(n = ", s$n[match(x, s$label)], ")")) +
    labs(x = NULL, y = measure$label, title = title,
         caption = wrap_caption(c(measure$note, notes,
                                  "Bars: mean of animals \u00b1 SEM; points: animals (mean of their sections).",
                                  test_note(stats, as.character(palette$label[palette$control]))), width = 62)) +
    theme_paper()
  if (artifacts) p <- p + artifact_shapes("animal") +
    theme(legend.position = "top", legend.text = element_text(size = 7))
  p
}

plot_by_animal <- function(sv, palette, measure, title = NULL, notes = character(), artifacts = FALSE) {
  a <- animal_values(sv)
  points <- if (artifacts) {
    geom_point(data = sv, aes(y = y, shape = artifacts), fill = "white", colour = "black", size = 1.5, stroke = 0.5,
               position = position_jitter(width = 0.15, height = 0, seed = 2))
  } else {
    geom_point(data = sv, aes(y = y), shape = 21, fill = "white", colour = "black", size = 1.5, stroke = 0.5,
               position = position_jitter(width = 0.15, height = 0, seed = 2))
  }
  p <- ggplot(a, aes(animal, y)) +
    geom_col(aes(fill = label), width = 0.75, colour = "black", linewidth = 0.35) +
    geom_errorbar(aes(ymin = y - sem, ymax = y + sem), width = 0.3, linewidth = 0.4) +
    points +
    scale_fill_manual(values = setNames(palette$colour, palette$label), guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
    facet_grid(. ~ label, scales = "free_x", space = "free_x") +
    labs(x = NULL, y = measure$label, title = title,
         caption = wrap_caption(c(measure$note, notes,
                                  "Bars: mean of the animal's sections \u00b1 SEM; points: sections."), width = 120)) +
    theme_paper() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7), strip.background = element_blank(),
          strip.text = element_text(face = "bold"))
  if (artifacts) p <- p + artifact_shapes("section") + theme(legend.position = "top")
  p
}

# PDF through the macOS Quartz device: it embeds Arial (cairo_pdf and pdf cannot find it here); elsewhere cairo_pdf.
pdf_device <- function(filename, width, height, ...) {
  if (capabilities("aqua")) grDevices::quartz(file = filename, type = "pdf", width = width, height = height)
  else grDevices::cairo_pdf(filename, width = width, height = height)
}

save_figure <- function(plot, out, width, height) {
  ggsave(paste0(out, ".pdf"), plot, width = width, height = height, units = "cm", device = pdf_device)
  ggsave(paste0(out, ".png"), plot, width = width, height = height, units = "cm", dpi = 300, device = ragg::agg_png)
}

# ---------------------------------------------------------------- tables behind the plots (to replot elsewhere)

# Animal values with one column per group (palette order) and one row per animal, shorter groups padded with NA:
# the layout of a Prism column table.
wide_by_group <- function(av, palette) {
  cols <- lapply(levels(palette$label), function(l) av$y[av$label == l])
  n <- max(lengths(cols))
  tibble::as_tibble(setNames(lapply(cols, function(v) c(v, rep(NA_real_, n - length(v)))), levels(palette$label)))
}

# Bar height and error bar of the by-genotype plots: mean of animals +- SEM (SD too).
group_summary <- function(av) {
  av |>
    group_by(label) |>
    summarise(n_animals = n(), mean = mean(y), sd = sd(y), sem = sd(y) / sqrt(n()), .groups = "drop") |>
    arrange(label)
}

# 5 significant digits; written with a UTF-8 byte-order mark so Excel shows special characters correctly
write_values <- function(df, out_dir, name) {
  df <- df |> mutate(across(where(is.double), \(v) signif(v, 5)))
  readr::write_excel_csv(df, file.path(out_dir, paste0(name, ".csv")), na = "")
}
