# The PSR statistics: Dunnett's test against the control directly on animal means (SPEC$anova_gate = FALSE), while the
# default stays gated on the one-way ANOVA (other assays rely on the default). Synthetic animals only.
library(testthat)
source(file.path("..", "plot_functions.R"), chdir = TRUE)
source(file.path("..", "psr", "measures.R"), chdir = TRUE)
source(file.path("..", "psr", "spec.R"), chdir = TRUE)
source("helper_palette.R")

# Group means 0, 0.8 and 0.3 with SD 1: the ANOVA is not significant (p >= 0.05), group A is still the largest shift.
not_significant <- function() {
  for (seed in 1:200) {
    av <- animals(c(0, 0.8, 0.3), n = c(4, 5, 5), sd = 1, seed = seed)
    fit <- summary(aov(y ~ group, data = av))[[1]][["Pr(>F)"]][1]
    if (fit > 0.1 && fit < 0.3) return(av)
  }
  stop("no seed gave a non-significant ANOVA")
}

test_that("the psr spec drops the ANOVA gate", {
  expect_false(SPEC$anova_gate)
})

test_that("without the gate, a non-significant ANOVA still gets its Dunnett result and says which test ran", {
  av <- not_significant()
  st <- dunnett_vs_control(av, palette, gate = !isFALSE(SPEC$anova_gate))
  expect_gte(st$anova_p[1], ALPHA)
  expect_true(all(st$posthoc))
  expect_false(anyNA(st$difference))
  expect_false(anyNA(st$p_adjusted))
  expect_equal(st$group, c("group_a", "group_b"))
  expect_equal(st$test, rep("Dunnett vs control, no ANOVA gate", 2))
  expect_match(test_note(st, "Control"), "Dunnett's test vs Control on animal means, no ANOVA first")
})

test_that("the default is gated: the same data give no post-hoc test and no test column", {
  st <- dunnett_vs_control(not_significant(), palette)
  expect_gte(st$anova_p[1], ALPHA)
  expect_false(any(st$posthoc))
  expect_true(all(is.na(st$p_adjusted)))
  expect_false("test" %in% names(st))
  expect_match(test_note(st, "Control"), "\\(not significant\\); no post-hoc test")
})

test_that("a significant ANOVA gives the same Dunnett numbers with and without the gate", {
  av <- animals(c(0, 4, 1), n = c(4, 5, 5), sd = 1, seed = 3)
  gated <- dunnett_vs_control(av, palette)
  open <- dunnett_vs_control(av, palette, gate = FALSE)
  expect_lt(gated$anova_p[1], ALPHA)
  expect_true(all(gated$posthoc))
  expect_equal(open |> dplyr::select(-test), gated, ignore_attr = TRUE)
  expect_equal(names(open), append(names(gated), "test", after = 1))
})
