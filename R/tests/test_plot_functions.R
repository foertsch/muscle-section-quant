# Run with: Rscript -e 'testthat::test_dir("R/tests")'  (from the repo folder)
library(testthat)
source(file.path("..", "plot_functions.R"), chdir = TRUE)
source("helper_palette.R")

test_that("stars follow the usual thresholds", {
  expect_equal(stars(c(0.00001, 0.0005, 0.005, 0.03, 0.2)), c("****", "***", "**", "*", "ns"))
})

test_that("Dunnett compares each group with the control only, after a significant ANOVA", {
  av <- animals(c(1, 30, 1))
  s <- dunnett_vs_control(av, palette)
  expect_setequal(s$group, c("group_a", "group_b"))
  expect_true(all(s$posthoc))
  expect_lt(s$p_adjusted[s$group == "group_a"], 0.001)
  expect_gt(s$p_adjusted[s$group == "group_b"], 0.05)
  expect_equal(s$n_control[1], 4)
  expect_equal(s$difference[s$group == "group_a"], mean(av$y[5:9]) - mean(av$y[1:4]), tolerance = 1e-8)
  expect_equal(s$detectable_difference[s$group == "group_a"], detectable_difference(s$residual_sd[1], 5, 4))
})

test_that("groups absent from the data are left out of the comparison", {
  av <- animals(c(1, 30, 1), n = c(4, 5, 0))
  s <- dunnett_vs_control(av, palette)
  expect_equal(s$group, "group_a")
})

test_that("no post-hoc test when the ANOVA is not significant", {
  s <- dunnett_vs_control(animals(c(1, 1, 1), seed = 2), palette)
  expect_gte(s$anova_p[1], ALPHA)
  expect_false(any(s$posthoc))
  expect_true(all(is.na(s$p_adjusted)) && all(is.na(s$stars)))
  expect_match(test_note(s), "no post-hoc test")
})

test_that("the detectable difference grows with the spread and shrinks with more animals", {
  expect_equal(detectable_difference(1, 5, 5), stats::power.t.test(n = 5, sd = 1, power = 0.8)$delta)
  expect_gt(detectable_difference(2, 5, 5), detectable_difference(1, 5, 5))
  expect_lt(detectable_difference(1, 10, 10), detectable_difference(1, 5, 5))
})

test_that("the test note names the control group", {
  s <- tibble::tibble(anova_p = 1e-5, posthoc = TRUE)
  expect_match(test_note(s, "Reference"), "Dunnett's test vs Reference\\.")
  expect_match(test_note(s), "vs control\\.")
})

test_that("section and animal values follow a measure, with an optional artifacts flag", {
  measure <- list(label = "x", note = "n", value = function(s) 100 * s$share)
  sections <- tibble::tibble(group = c("control", "control", "group_a", "group_a", "other"),
                             animal = c("a", "a", "b", "b", "z"), share = c(0.01, 0.03, 0.4, 0.5, 9))
  sv <- section_values(sections, palette, measure)
  expect_equal(sv$y, c(1, 3, 40, 50))
  expect_equal(as.character(sv$label), c("Control", "Control", "Group A", "Group A"))
  av <- animal_values(sv)
  expect_equal(av$y[av$animal == "a"], 2)
  expect_equal(av$n_sections[av$animal == "a"], 2)
  expect_false("artifacts" %in% names(av))
  av2 <- animal_values(sv |> mutate(artifacts = c(TRUE, FALSE, FALSE, FALSE)))
  expect_true(av2$artifacts[av2$animal == "a"])   # one of two sections flagged
  expect_false(av2$artifacts[av2$animal == "b"])
  one_muscle <- section_values(sections |> mutate(muscle = c("Q", "T", "Q", "Q", "Q")), palette, measure,
                               keep = c("Q", "T", "Q", "Q", "Q") == "Q")
  expect_equal(one_muscle$y, c(1, 40, 50))
})

test_that("wide_by_group puts each group in its own column, padded with NA", {
  av <- animals(c(1, 30, 20), n = c(2, 3, 1)) |> with_labels(palette)
  w <- wide_by_group(av, palette)
  expect_equal(names(w), c("Control", "Group A", "Group B"))
  expect_equal(nrow(w), 3)
  expect_true(is.na(w[["Control"]][3]) && is.na(w[["Group B"]][2]))
})

test_that("group_summary gives the bar height and SEM of the by-genotype plot", {
  av <- tibble::tibble(group = c("control", "control", "group_a"), animal = c("1", "2", "3"), y = c(1, 3, 10)) |>
    with_labels(palette)
  g <- group_summary(av)
  expect_equal(g$mean[g$label == "Control"], 2)
  expect_equal(g$sem[g$label == "Control"], sd(c(1, 3)) / sqrt(2))
})

test_that("the plots build with and without artifacts and a title", {
  av <- animals(c(1, 30, 1)) |> with_labels(palette)
  s <- dunnett_vs_control(av, palette)
  measure <- list(label = "x", note = "n", value = function(s) s$y)
  expect_s3_class(plot_by_genotype(av, palette, s, measure, title = "QUAD", notes = "a note"), "ggplot")
  sv <- av |> mutate(artifacts = FALSE)
  expect_s3_class(plot_by_genotype(av |> mutate(artifacts = FALSE), palette, s, measure, artifacts = TRUE), "ggplot")
  expect_s3_class(plot_by_animal(sv, palette, measure, artifacts = TRUE), "ggplot")
  expect_s3_class(plot_by_animal(sv, palette, measure), "ggplot")
})

test_that("the plotting code is plain ASCII, so labels render in any locale", {
  sources <- list.files("..", pattern = "\\.R$", recursive = TRUE, full.names = TRUE)
  for (f in sources[!grepl("/tests/", sources)]) {
    bytes <- readBin(f, "raw", n = 1e6)
    expect_true(all(as.integer(bytes) < 128), info = f)
  }
})
