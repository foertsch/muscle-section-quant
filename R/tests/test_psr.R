library(testthat)
source(file.path("..", "plot_functions.R"), chdir = TRUE)
source(file.path("..", "psr", "measures.R"), chdir = TRUE)
source("helper_palette.R")

test_that("animals average their sections and are marked when half their sections have freezing artifacts", {
  sections <- tibble::tibble(group = c("control", "control", "control", "group_a", "group_a"),
                             animal = c("a", "a", "b", "c", "c"),
                             red_fraction = c(0.1, 0.12, 0.09, 0.2, 0.3), collagen_outside_fraction = c(0.05, 0.07, 0.04, 0.1, 0.2),
                             n_fibers = c(100, 300, 200, 50, 150),
                             section_area_mm2 = c(10, 10, 20, 5, 5), freezing_artifact_area = c(0.8, 0.1, 0, 0, NA))
  sv <- psr_section_values(sections, palette, "fiber_density")
  expect_equal(sv$y, c(10, 30, 10, 10, 30))
  av <- animal_values(sv)
  expect_equal(av$y[av$animal == "a"], 20)
  expect_true(av$artifacts[av$animal == "a"])    # one of two sections affected
  expect_false(av$artifacts[av$animal == "c"])   # NA counts as not affected
  col <- animal_values(psr_section_values(sections, palette, "collagen_outside"))
  expect_equal(col$y[col$animal == "a"], 6)
})

test_that("a run without a hole survey has no freezing-artifact column and marks nothing", {
  sections <- tibble::tibble(group = c("control", "group_a"), animal = c("a", "c"),
                             collagen_outside_fraction = c(0.05, 0.1), n_fibers = c(100, 50),
                             section_area_mm2 = c(10, 5))
  sv <- psr_section_values(sections, palette, "collagen_outside")
  expect_false("artifacts" %in% names(sv))
  expect_equal(animal_values(sv)$y, c(5, 10))
})

test_that("each animal counts once in the size densities", {
  fibers <- tibble::tibble(group = "control", animal = rep(c("a", "b"), c(5000, 50)),
                           min_feret_um = c(rnorm(5000, 30, 2), rnorm(50, 60, 2)))
  d <- size_densities(fibers, palette, "min_feret_um")
  left <- sum(d$mean[d$x < 45]); right <- sum(d$mean[d$x >= 45])
  expect_equal(left, right, tolerance = 0.1)
})

test_that("density_table has x, then mean and SEM per group", {
  set.seed(1)
  fibers <- tibble::tibble(group = rep(c("control", "group_a"), each = 400),
                           animal = rep(c("a", "b", "c", "d"), each = 200),
                           min_feret_um = c(rnorm(400, 45, 8), rnorm(400, 30, 8)))
  d <- density_table(fibers, palette, "min_feret_um")
  expect_equal(names(d), c("x", "Control mean", "Control sem", "Group A mean", "Group A sem"))
  expect_equal(nrow(d), 512)
  expect_equal(sum(d[["Control mean"]]) * diff(d$x[1:2]), 1, tolerance = 0.02)   # a density integrates to ~1
})

test_that("the bar measures read their columns and a size density is the kernel density at the fixed bandwidth", {
  s <- tibble::tibble(group = "control", animal = "a", red_fraction = 0.1, collagen_outside_fraction = 0.04,
                      n_fibers = 100, section_area_mm2 = 10, freezing_artifact_area = 0.8)
  expect_equal(c(psr_section_values(s, palette, "collagen_outside")$y, psr_section_values(s, palette, "fiber_count")$y), c(4, 100))
  expect_false("fibrosis" %in% names(BAR_MEASURES))
  expect_match(BAR_MEASURES$collagen_outside$label, "^Collagen outside the fiber outlines \\(%\\)$")
  expect_match(BAR_MEASURES$fiber_count$label, "exploratory")
  expect_match(BAR_MEASURES$fiber_density$label, "exploratory")
  set.seed(2)
  v <- rnorm(300, 45, 8)
  d <- size_densities(tibble::tibble(group = "control", animal = "a", min_feret_um = v), palette, "min_feret_um")
  expect_equal(d$mean, density(v, bw = 2, from = 0, to = 100, n = 512)$y)
})
