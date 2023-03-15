# Explanations

## Methods

This section lists all currently implemented model explanation backends in CaTabRa.
Refer to Section "Extending the CaTabRa Explanation Framework" for information on how to integrate other backends.

### SHAP

SHAP [1] is the default model explanation backend used by CaTabRa.
Set config parameter `"explainer"` to `"shap"` to enable it (it is enabled by default anyway).

SHAP explains models in terms of *local feature importance scores*. That means, for every sample a scalar importance
value is assigned to each feature. These values reflect how much each feature contributes to the final model output in
the given sample, both for *increasing* and for *decreasing* it. Hence, importance values can be positive and negative.
Furthermore, they usually differ between samples.
Global model explanations computed with SHAP simply correspond to average local importance values across a given data
set, where the average is computed separately for positive and negative contributions.

Ensembles are explained by explaining their constituent pipelines individually. In classification tasks, care must be
taken when combining individual explanations into "overall" explanations of the whole ensemble (see "Interpretation of
Results" for the reason), which is why this is not done automatically. Better compare individual explanations
qualitatively to get insights into which features are important for the ensemble.

Pipelines are explained following the methodology outlined in Section "Explaining Pipelines with Model-Specific
Methods". If this is not possible because importance cannot be back-propagated through one of the preprocessing steps,
the pipeline is explained as a whole using SHAP's `KernelExplainer`. Note that this might be very slow.

#### Interpretation of Results

In theory, SHAP values satisfy the so-called *local accuracy* property (a.k.a. sum-to-delta property). That means, for a
given sample *x* the sum of the feature importance values of *x* equals the model output at *x*, minus the expected
(average) model output. The importance of a feature therefore directly reflects how much the corresponding feature
contributes to the model output, both positively (if the importance is positive) and negatively (if the importance is
negative), compared to the average output. This nice property renders SHAP an appealing method for determining
feature importance. Please note that "model output" not necessarily refers to class probabilities in classification
problems; depending on the actual model type, it might also refer to "pre-logit" log-odds values. This is why
explanations from different models cannot easily be combined, if some refer to class probabilities and others to
log-odds.

There are two important points to consider when interpreting feature importance scores calculated by CaTabRa using SHAP:
1. The [shap package](https://github.com/slundberg/shap) [2] used internally usually approximates exact SHAP values for
    reasons of efficiency, which means that local accuracy might not always be satisfied. Fast exact methods only exist
    for some model types, like tree-based models.
2. When explaining a pipeline as per the methodology outlined in Section "Explaining Pipelines with Model-Specific
    Methods", feature importance values are first computed for the actual estimator and then back-propagated through
    all the preprocessing steps. *This back-propagation might not preserve the characteristic properties of SHAP values,
    in particular local accuracy.* This is especially true for (quasi) linear transformations (PCA, ICA, ...).

#### References

[1] Scott M Lundberg and Su-In Lee.
[*A Unified Approach to Interpreting Model Predictions*](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf).
Advances in Neural Information Processing Systems 30: 4765–4774, 2017.

[2] [shap Github repository](https://github.com/slundberg/shap)

### Permutation Importance

Permutation importance [1] is a simple model-agnostic approach to calculate global feature importance scores, without
taking potential feature interactions into account.
Since it does not need to be fit to training data, there is no need to adjust config parameter `"explainer"`. Simply
pass `--explainer permutation` when invoking command `explain`, or set the keyword argument `explainer="permutation"`
of function `catabra.explanation.explain()`.

Permutation importance is calculated by randomly permuting a feature and comparing the performance of a prediction
model to its performance on the original data, wrt. some given metric. The performance drop can be regarded as an
indicator of the importance of the feature. This process is repeated for all features individually, i.e., only one
single feature is permuted at a time. More information, also regarding possible caveats, can be found in the
[scikit-learn documentation](https://scikit-learn.org/stable/modules/permutation_importance.html).

Permutation importance can only produce *global* feature importance scores and needs *labeled data*. Furthermore, it
does not distinguish between positive and negative contributions, as typically (though not necessarily always) the
calculated importance scores will be non-negative.

Since permutation importance is inherently model-agnostic, it is always applied to full pipelines at once. Thus,
importance scores are never back-propagated through preprocessing steps, as described in Section "Explaining Pipelines
with Model-Specific Methods".

#### Interpretation of Results

The importance of a feature *f* reflects the performance drop of the model to explain when *f* is "switched off", which
is accomplished by randomly permuting *f*. Hence, the importance not only depends on the model and the data it is
explained on, but also on the metric used for measuring the performance.

In contrast to more sophisticated model explanation techniques, permutation importance does not take feature
interactions into account. Thus, if two features carrying useful information are strongly correlated, permutation
importance might deem both unimportant (depending on the model).

#### References

[1] Leo Breiman. [*Random Forests*](https://doi.org/10.1023/A:1010933404324). Machine Learning 45(1): 5-32, 2001.

## Explaining Pipelines with Model-Specific Methods

A pipeline consists of a (possibly empty) list of preprocessing steps and a final estimator. Explaining a pipeline can
be accomplished in two ways:
1. Apply model-agnostic explanation methods to the whole pipeline at once.
2. Apply model-specific explanation methods to the final estimator and "back-propagate" feature importance values thus
    obtained through the individual preprocessing steps.

Model-specific explanation methods are generally preferable over model-agnostic ones *if they are applicable to a given
estimator*, either because they are faster to compute or because they return more accurate results. Therefore, CaTabRa
by default attempts to pursue Approach 2 and only falls back to Approach 1 if Approach 2 fails, provided the selected
explanation backend has a model-agnostic method in its portfolio.

For a given input *x*, explaining the behavior of a pipeline on *x* following Approach 2 consists of three phases:
1. Apply the preprocessing steps to *x* to obtain the processed input *x'* (forward pass).
2. Explain the estimator on *x'*; call the resulting explanation *s'*.
3. Back-propagate *s'* through the preprocessing steps to obtain final explanation *s* (backward pass).

(Note the conceptual similarity to *Layer-wise Relevance Propagation* [1] for explaining deep neural networks.)

The interesting part is the last phase, since back-propagating feature importance through preprocessing transformations
is intricate. At present, there exists no general solution, but only specific implementations for some of the most
widely used transformations; Section "Extending the CaTabRa Explanation Framework" sketches how implementations for
other transformations can be added.

The currently implemented back-propagation policy adheres to the following general rules:
* Transformations that act on each feature individually are regarded as identity transformations during
    back-propagation, i.e., feature importance values are back-propagated unchanged. Transformations of this kind
    include all sorts of feature scaling (standardization, normalization, etc.).
* If a transformation produces multiple output features from a single input feature, the importance values of the
    output features are summed to obtain the importance of the input feature; a typical example for this kind of
    transformation is one-hot encoding of categorical variables.
* Transformations that drop features assign zero importance to these features upon back-propagation. These include all
    sorts of feature selection methods.
* Linear and quasi-linear transformations are treated uniformly by distributing importance relative to the magnitude of
    the coefficients of the respective input features; the actual feature values of the sample under consideration are
    *not* taken into account. Linear transformations include famous PCA and ICA, and quasi-linear transformations are
    like linear transformations except that feature-wise functions are applied before or after the actual linear
    transformation; an example of a quasi-linear transformation is `sklearn.kernel_approximation.RBFSampler`.<br>
    **Note**: There are alternative approaches to dealing with (quasi-)linear transformations, see the discussion in
    `catabra.explanation.sklearn_explainer._LinearTransformationExplainer` for details.

Finally, there are transformations through which feature importance cannot be back-propagated. Nystroem transformations
[2], for instance, completely alter the feature space by constructing an approximate feature map for an arbitrary
kernel using a subset of the training data as basis.

#### References

[1] Gregoire Montavon, Alexander Binder et al.
[Layer-Wise Relevance Propagation: An Overview](https://doi.org/10.1007/978-3-030-28954-6_10).
In *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*, pp. 193-209,
Lecture Notes in Computer Science, 2019.

[2] [scikit-learn Nystroem transformation](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem)

## Extending the CaTabRa Explanation Framework

### Adding New Explanation Backends

New explanation backends can be added by implementing the abstract base class
`catabra.explanation.base.EnsembleExplainer` and then registering the subclass with the `register()` method of the base
class. In doing so, the new backend can be used just like the built-in shap backend; in particular, it can be activated
by simply setting the value of config parameter `"explainer"` to its name.

The built-in shap backend serves as a role model for how new backends can be added. The abstract base class is
implemented by `catabra.explanation._shap.backend.SHAPEnsembleExplainer`, which is registered in
`catabra.explanation._shap.__init__`.

### Adding New Back-Propagation Rules for Transformations

Adding new back-Propagation rules for transformations amounts to implementing abstract base class
`catabra.explanation.base.TransformationExplainer` and registering a factory with the `register_factory()` method of
the base class.

The built-in backpropagation rules for some of the most widespread sklearn transformations serve as a role model.
Refer to `catabra.explanation.sklearn_explainer`, where first `sklearn_explainer_factory()` is defined and
then `TransformationExplainer` is implemented for each of the transformations considered; the factory simply returns
an instance of the appropriate subclass depending on the type of parameter `obj`. The factory is finally registered in
`catabra.explanation.__init__`.