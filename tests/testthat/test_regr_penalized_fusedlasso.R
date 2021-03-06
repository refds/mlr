context("regr_penalized_fusedlasso")

test_that("regr_penalized_fusedlasso", {
  requirePackages("!penalized", default.method = "load")
  
  parset.list = list(
    list(),
    list(maxiter = 2L),
    list(lambda1 = 2, maxiter = 4L),
    list(lambda1 = 2, lambda2 = 1, maxiter = 2L),
    list(lambda2 = 2, maxiter = 4L)
  )

  # to make test of empty list feasable (in terms of time), number of obs need to be reduced
  regr.train.inds = sample(seq(1,506), size = 150)
  regr.test.inds  = setdiff(1:nrow(regr.df), regr.train.inds)
  regr.train = regr.df[regr.train.inds, ]
  regr.test  = regr.df[regr.test.inds, ]
  
  old.predicts.list = list()
  old.probs.list = list()

  for (i in 1:length(parset.list)) {
    parset = parset.list[[i]]
    if (is.null(parset$lambda1))
      parset$lambda1 = 1
    if (is.null(parset$lambda2))
      parset$lambda2 = 1
    pars = list(regr.formula, data = regr.train, fusedl = TRUE, trace = FALSE)
    pars = c(pars, parset)
    set.seed(getOption("mlr.debug.seed"))
    capture.output(
      m <- do.call(penalized::penalized, pars)
    )
    # FIXME: should be removed, reported in issue 840
    m@formula$unpenalized[[2L]] = as.symbol(regr.target)
    p = penalized::predict(m, data = regr.test)
    old.predicts.list[[i]] = p[,"mu"]
  }
  testSimpleParsets("regr.penalized.fusedlasso", regr.df, regr.target,
    regr.train.inds, old.predicts.list, parset.list)

  parset.list = list(
    list(),
    list(lambda1 = 2, lambda2 = 1, maxiter = 2L),
    list(lambda1 = 1, lambda2 = 2, maxiter = 4L)
  )

  tt = function(formula, data, subset = 1:nrow(data), ...) {
    args = list(...)
    if (is.null(args$lambda1) & is.null(args$lambda2)) {
      penalized::penalized(formula, data = data[subset, ],
        fusedl = TRUE, trace = FALSE, lambda1 = 1, lambda2 = 1, ...)
    } else {
      penalized::penalized(formula, data = data[subset, ],
        fusedl = TRUE, trace = FALSE, ...)
    }
  }

  tp = function(model, newdata, ...) {
    penalized::predict(model, data = newdata,...)[, "mu"]
  }

  testCVParsets("regr.penalized.fusedlasso", regr.df, regr.target,
    tune.train = tt, tune.predict = tp, parset.list = parset.list)
})
