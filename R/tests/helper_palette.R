# Generic groups for the tests (the lab's cohorts live in untracked local files).
palette <- tibble::tibble(group = c("control", "group_a", "group_b"),
                          label = factor(c("Control", "Group A", "Group B"), levels = c("Control", "Group A", "Group B")),
                          colour = c("#898781", "#2a78d6", "#eb6834"), control = c(TRUE, FALSE, FALSE))

animals <- function(means, n = c(4, 5, 5), sd = 1, seed = 1) {
  set.seed(seed)
  tibble::tibble(group = rep(c("control", "group_a", "group_b"), n), animal = as.character(seq_len(sum(n))),
                 y = unlist(Map(function(m, k) rnorm(k, m, sd), means, n)))
}
