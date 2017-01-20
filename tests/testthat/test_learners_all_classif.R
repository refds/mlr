context("learners_all_classif1")

test_that("learners work: classif1 ", {

  # settings to make learners faster and deal with small data size
  hyperpars = list(
    classif.boosting = list(mfinal = 2L),
    classif.cforest = list(mtry = 1L),
    classif.bartMachine = list(verbose = FALSE, run_in_sample = FALSE,
      # without this (and despite use_missing_data being TRUE), the test with missing data fails with a null point exception, which manifests itself as a completely different rJava error in the test
      replace_missing_data_with_x_j_bar = TRUE,
      num_iterations_after_burn_in = 10L),
    classif.bdk = list(ydim = 2L),
    classif.earth = list(degree = 3L, nprune = 2L),
    classif.gbm = list(bag.fraction = 1, n.minobsinnode = 1),
    classif.lssvm = list(kernel = "rbfdot", reduced = FALSE),
    classif.nodeHarvest = list(nodes = 100L, nodesize = 5L),
    classif.xyf = list(ydim = 2L),
    classif.h2o.deeplearning = list(hidden = 2L, seed = getOption("mlr.debug.seed"), reproducible = TRUE),
    classif.h2o.randomForest = list(seed = getOption("mlr.debug.seed"))
  )


  # binary classif
  task = subsetTask(binaryclass.task, subset = c(10:20, 180:190),
    features = getTaskFeatureNames(binaryclass.task)[12:15])

  #lrns = mylist(task, create = TRUE)
  #lapply(lrns, testThatLearnerParamDefaultsAreInParamSet)
  #lapply(lrns, testBasicLearnerProperties, task = task, hyperpars = hyperpars)

  ######
  # NOTE: properties = "numeric" and "twoclass" cause the DLL error
  ######
  # Testing Binary with all []:
  # numerics [factors] [ordered] [missings] [weights] [prob] [oneclass] twoclass [multiclass]

  # oneclass
  lrns = mylist(task, create = TRUE, properties = "oneclass")
  lapply(lrns, testThatLearnerParamDefaultsAreInParamSet)
  lapply(lrns, testBasicLearnerProperties, task = task, hyperpars = hyperpars)

  # multiclass
  lrns = mylist(task, create = TRUE, properties = "multiclass")
  lapply(lrns, testThatLearnerParamDefaultsAreInParamSet)
  lapply(lrns, testBasicLearnerProperties, task = task, hyperpars = hyperpars)

  # binary classif with factors
  lrns = mylist("classif", properties = "factors", create = TRUE)
  lapply(lrns, testThatLearnerHandlesFactors, task = task, hyperpars = hyperpars)

  # binary classif with ordered factors
  lrns = mylist("classif", properties = "ordered", create = TRUE)
  lapply(lrns, testThatLearnerHandlesOrderedFactors, task = task, hyperpars = hyperpars)
  detachAllPackages()
  R.utils::gcDLLs()
})


test_that("weightedClassWrapper on all binary learners",  {
  R.utils::gcDLLs()
  pos = getTaskDescription(binaryclass.task)$positive
  f = function(lrn, w) {
    lrn1 = makeLearner(lrn)
    lrn2 = makeWeightedClassesWrapper(lrn1, wcw.weight = w)
    m = train(lrn2, binaryclass.task)
    p = predict(m, binaryclass.task)
    cm = calculateConfusionMatrix(p)$result
  }

  learners = listLearners(binaryclass.task, "class.weights")
  x = lapply(learners$class, function(lrn) {
    cm1 = f(lrn, 0.001)
    cm2 = f(lrn, 1)
    cm3 = f(lrn, 1000)
    expect_true(all(cm1[, pos] <= cm2[, pos]))
    expect_true(all(cm2[, pos] <= cm3[, pos]))
  })
  R.utils::gcDLLs()
})


test_that("WeightedClassWrapper on all multiclass learners",  {
  R.utils::gcDLLs()
  levs = getTaskClassLevels(multiclass.task)
  f = function(lrn, w) {
    lrn1 = makeLearner(lrn)
    param = lrn1$class.weights.param
    lrn2 = makeWeightedClassesWrapper(lrn1, wcw.weight = w)
    m = train(lrn2, multiclass.task)
    p = predict(m, multiclass.task)
    cm = calculateConfusionMatrix(p)$result
  }

  learners = listLearners(multiclass.task, "class.weights")
  x = lapply(learners$class, function(lrn) {
    classes = getTaskFactorLevels(multiclass.task)[[multiclass.target]]
    n = length(classes)
    cm1 = f(lrn, setNames(object = c(10000, 1, 1), classes))
    cm2 = f(lrn, setNames(object = c(1, 10000, 1), classes))
    cm3 = f(lrn, setNames(object = c(1, 1, 10000), classes))
    expect_true(all(cm1[, levs[1]] >= cm2[, levs[1]]))
    expect_true(all(cm1[, levs[1]] >= cm3[, levs[1]]))
    expect_true(all(cm2[, levs[2]] >= cm1[, levs[2]]))
    expect_true(all(cm2[, levs[2]] >= cm3[, levs[2]]))
    expect_true(all(cm3[, levs[3]] >= cm1[, levs[3]]))
    expect_true(all(cm3[, levs[3]] >= cm2[, levs[3]]))
  })
  detachAllPackages()
  R.utils::gcDLLs()
})
