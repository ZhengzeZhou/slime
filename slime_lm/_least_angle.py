"""
Least Angle Regression algorithm. See the documentation on the
Generalized Linear Model for a complete discussion.
"""
# Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux
#
# License: BSD 3 clause

from math import log
import sys
import warnings

import numpy as np
from scipy import linalg, interpolate
from scipy.linalg.lapack import get_lapack_funcs
from scipy import stats
from joblib import Parallel

# mypy error: Module 'sklearn.utils' has no attribute 'arrayfuncs'
from sklearn.utils import arrayfuncs, as_float_array  # type: ignore
from sklearn.exceptions import ConvergenceWarning

SOLVE_TRIANGULAR_ARGS = {'check_finite': False}


def lars_path(
    X,
    y,
    Xy=None,
    *,
    Gram=None,
    max_iter=500,
    alpha_min=0,
    method="lar",
    copy_X=True,
    eps=np.finfo(float).eps,
    copy_Gram=True,
    verbose=0,
    return_path=True,
    return_n_iter=False,
    positive=False,
    testing=False,
    alpha=0.05,
    testing_stop=False,
    testing_verbose=False,
):
    """Compute Least Angle Regression or Lasso path using LARS algorithm [1]
    The optimization objective for the case method='lasso' is::
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])
    Read more in the :ref:`User Guide <least_angle_regression>`.
    Parameters
    ----------
    X : None or array-like of shape (n_samples, n_features)
        Input data. Note that if X is None then the Gram matrix must be
        specified, i.e., cannot be None or False.
    y : None or array-like of shape (n_samples,)
        Input targets.
    Xy : array-like of shape (n_samples,) or (n_samples, n_targets), \
            default=None
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.
    Gram : None, 'auto', array-like of shape (n_features, n_features), \
            default=None
        Precomputed Gram matrix (X' * X), if ``'auto'``, the Gram
        matrix is precomputed from the given X, if there are more samples
        than features.
    max_iter : int, default=500
        Maximum number of iterations to perform, set to infinity for no limit.
    alpha_min : float, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.
    method : {'lar', 'lasso'}, default='lar'
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.
    copy_X : bool, default=True
        If ``False``, ``X`` is overwritten.
    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.
    copy_Gram : bool, default=True
        If ``False``, ``Gram`` is overwritten.
    verbose : int, default=0
        Controls output verbosity.
    return_path : bool, default=True
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.
    return_n_iter : bool, default=False
        Whether to return the number of iterations.
    positive : bool, default=False
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.
    testing : bool, default=False
        Whether to conduct hypothesis testing each time a new variable enters
    alpha : float, default=0.05
        Significance level of hypothesis testing. Valid only if testing is True.
    testing_stop : bool, default=False
        If set to True, stops calculating future paths when the test yields
        insignificant results.
        Only takes effect when testing is set to True.
    testing_verbose : bool, default=True
        Controls output verbosity for hypothese testing procedure. 
    Returns
    -------
    alphas : array-like of shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller.
    active : array-like of shape (n_alphas,)
        Indices of active variables at the end of the path.
    coefs : array-like of shape (n_features, n_alphas + 1)
        Coefficients along the path
    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.
    test_result: disctionary
        Contains testing results in the form of [test_stats, new_n] produced 
        at each step. Returned only if testing is set to True.
    See Also
    --------
    lars_path_gram
    lasso_path
    lasso_path_gram
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode
    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf
    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_
    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_
    """
    if X is None and Gram is not None:
        raise ValueError(
            'X cannot be None if Gram is not None'
            'Use lars_path_gram to avoid passing X and y.'
        )
    return _lars_path_solver(
        X=X, y=y, Xy=Xy, Gram=Gram, n_samples=None, max_iter=max_iter,
        alpha_min=alpha_min, method=method, copy_X=copy_X,
        eps=eps, copy_Gram=copy_Gram, verbose=verbose, return_path=return_path,
        return_n_iter=return_n_iter, positive=positive, testing=testing, 
        alpha=alpha, testing_stop=testing_stop, testing_verbose=testing_verbose)


def lars_path_gram(
    Xy,
    Gram,
    *,
    n_samples,
    max_iter=500,
    alpha_min=0,
    method="lar",
    copy_X=True,
    eps=np.finfo(float).eps,
    copy_Gram=True,
    verbose=0,
    return_path=True,
    return_n_iter=False,
    positive=False
):
    """lars_path in the sufficient stats mode [1]
    The optimization objective for the case method='lasso' is::
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])
    Read more in the :ref:`User Guide <least_angle_regression>`.
    Parameters
    ----------
    Xy : array-like of shape (n_samples,) or (n_samples, n_targets)
        Xy = np.dot(X.T, y).
    Gram : array-like of shape (n_features, n_features)
        Gram = np.dot(X.T * X).
    n_samples : int or float
        Equivalent size of sample.
    max_iter : int, default=500
        Maximum number of iterations to perform, set to infinity for no limit.
    alpha_min : float, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.
    method : {'lar', 'lasso'}, default='lar'
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.
    copy_X : bool, default=True
        If ``False``, ``X`` is overwritten.
    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.
    copy_Gram : bool, default=True
        If ``False``, ``Gram`` is overwritten.
    verbose : int, default=0
        Controls output verbosity.
    return_path : bool, default=True
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.
    return_n_iter : bool, default=False
        Whether to return the number of iterations.
    positive : bool, default=False
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.
    Returns
    -------
    alphas : array-like of shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller.
    active : array-like of shape (n_alphas,)
        Indices of active variables at the end of the path.
    coefs : array-like of shape (n_features, n_alphas + 1)
        Coefficients along the path
    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.
    See Also
    --------
    lars_path
    lasso_path
    lasso_path_gram
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode
    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf
    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_
    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_
    """
    return _lars_path_solver(
        X=None, y=None, Xy=Xy, Gram=Gram, n_samples=n_samples,
        max_iter=max_iter, alpha_min=alpha_min, method=method,
        copy_X=copy_X, eps=eps, copy_Gram=copy_Gram,
        verbose=verbose, return_path=return_path,
        return_n_iter=return_n_iter, positive=positive)


def _lars_path_solver(
    X,
    y,
    Xy=None,
    Gram=None,
    n_samples=None,
    max_iter=500,
    alpha_min=0,
    method="lar",
    copy_X=True,
    eps=np.finfo(float).eps,
    copy_Gram=True,
    verbose=0,
    return_path=True,
    return_n_iter=False,
    positive=False,
    testing=False,
    alpha=0.05,
    testing_stop=False,
    testing_verbose=False,
):
    """Compute Least Angle Regression or Lasso path using LARS algorithm [1]
    The optimization objective for the case method='lasso' is::
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])
    Read more in the :ref:`User Guide <least_angle_regression>`.
    Parameters
    ----------
    X : None or ndarray of shape (n_samples, n_features)
        Input data. Note that if X is None then Gram must be specified,
        i.e., cannot be None or False.
    y : None or ndarray of shape (n_samples,)
        Input targets.
    Xy : array-like of shape (n_samples,) or (n_samples, n_targets), \
            default=None
        `Xy = np.dot(X.T, y)` that can be precomputed. It is useful
        only when the Gram matrix is precomputed.
    Gram : None, 'auto' or array-like of shape (n_features, n_features), \
            default=None
        Precomputed Gram matrix `(X' * X)`, if ``'auto'``, the Gram
        matrix is precomputed from the given X, if there are more samples
        than features.
    n_samples : int or float, default=None
        Equivalent size of sample. If `None`, it will be `n_samples`.
    max_iter : int, default=500
        Maximum number of iterations to perform, set to infinity for no limit.
    alpha_min : float, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.
    method : {'lar', 'lasso'}, default='lar'
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.
    copy_X : bool, default=True
        If ``False``, ``X`` is overwritten.
    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.
    copy_Gram : bool, default=True
        If ``False``, ``Gram`` is overwritten.
    verbose : int, default=0
        Controls output verbosity.
    return_path : bool, default=True
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.
    return_n_iter : bool, default=False
        Whether to return the number of iterations.
    positive : bool, default=False
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.
    testing : bool, default=False
        Whether to conduct hypothesis testing each time a new variable enters
    alpha : float, default=0.05
        Significance level of hypothesis testing. Valid only if testing is True.
    testing_stop : bool, default=False
        If set to True, stops calculating future paths when the test yields
        insignificant results.
        Only takes effect when testing is set to True.
    testing_verbose : bool, default=True
        Controls output verbosity for hypothese testing procedure.
    Returns
    -------
    alphas : array-like of shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller.
    active : array-like of shape (n_alphas,)
        Indices of active variables at the end of the path.
    coefs : array-like of shape (n_features, n_alphas + 1)
        Coefficients along the path
    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.
    test_result: dictionary
        Contains testing results in the form of [test_stats, new_n] produced 
        at each step. Returned only if testing is set to True.
    See Also
    --------
    lasso_path
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode
    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf
    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_
    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_
    """
    if method == "lar" and positive:
        raise ValueError(
            "Positive constraint not supported for 'lar' " "coding method."
        )

    n_samples = n_samples if n_samples is not None else y.size

    if Xy is None:
        Cov = np.dot(X.T, y)
    else:
        Cov = Xy.copy()

    if Gram is None or Gram is False:
        Gram = None
        if X is None:
            raise ValueError('X and Gram cannot both be unspecified.')
    elif isinstance(Gram, str) and Gram == 'auto' or Gram is True:
        if Gram is True or X.shape[0] > X.shape[1]:
            Gram = np.dot(X.T, X)
        else:
            Gram = None
    elif copy_Gram:
        Gram = Gram.copy()

    if Gram is None:
        n_features = X.shape[1]
    else:
        n_features = Cov.shape[0]
        if Gram.shape != (n_features, n_features):
            raise ValueError('The shapes of the inputs Gram and Xy'
                             ' do not match.')

    if copy_X and X is not None and Gram is None:
        # force copy. setting the array to be fortran-ordered
        # speeds up the calculation of the (partial) Gram matrix
        # and allows to easily swap columns
        X = X.copy('F')

    max_features = min(max_iter, n_features)

    dtypes = set(a.dtype for a in (X, y, Xy, Gram) if a is not None)
    if len(dtypes) == 1:
        # use the precision level of input data if it is consistent
        return_dtype = next(iter(dtypes))
    else:
        # fallback to double precision otherwise
        return_dtype = np.float64

    if return_path:
        coefs = np.zeros((max_features + 1, n_features), dtype=return_dtype)
        alphas = np.zeros(max_features + 1, dtype=return_dtype)
    else:
        coef, prev_coef = (np.zeros(n_features, dtype=return_dtype),
                           np.zeros(n_features, dtype=return_dtype))
        alpha, prev_alpha = (np.array([0.], dtype=return_dtype),
                             np.array([0.], dtype=return_dtype))
        # above better ideas?

    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization. Only lower part is
    # referenced.
    if Gram is None:
        L = np.empty((max_features, max_features), dtype=X.dtype)
        swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (X,))
    else:
        L = np.empty((max_features, max_features), dtype=Gram.dtype)
        swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (Cov,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (L,))

    if verbose:
        if verbose > 1:
            print("Step\t\tAdded\t\tDropped\t\tActive set size\t\tC")
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning
    equality_tolerance = np.finfo(np.float32).eps

    residual = y - 0
    coef = np.zeros(n_features)
    test_result = {}

    if Gram is not None:
        Gram_copy = Gram.copy()
        Cov_copy = Cov.copy()

    z_score = stats.norm.ppf(1 - alpha)
    while True:
        if not testing:
            if Cov.size:
                if positive:
                    C_idx = np.argmax(Cov)
                else:
                    C_idx = np.argmax(np.abs(Cov))

                C_ = Cov[C_idx]

                if positive:
                    C = C_
                else:
                    C = np.fabs(C_)
            else:
                C = 0.
        else:
            # not implemented when if positive is set to True
            if Cov.size:
                if positive:
                    C_idx = np.argmax(Cov)
                else:
                    C_idx = np.argmax(np.abs(Cov))
                    if Cov.size > 1:
                        C_idx_second = np.abs(Cov).argsort()[-2]

                        x1 = X.T[n_active + C_idx]
                        x2 = X.T[n_active + C_idx_second]

                        residual = y - np.dot(X[:, :n_active], coef[active])
                        u = np.array([np.dot(x1, residual), np.dot(x2, residual)]) / len(y) 
                        cov = np.cov(x1 * residual, x2 * residual)

                        new_n = len(y)
                        if u[0] >= 0 and u[1] >= 0:
                            test_stats = u[0] - u[1] - z_score * np.sqrt(2 * (cov[0][0] + cov[1][1] - cov[0][1] - cov[1][0]) / len(y))
                            if test_stats < 0:
                                z_alpha = (u[0] - u[1]) / np.sqrt(2 * (cov[0][0] + cov[1][1] - cov[0][1] - cov[1][0]) / len(y))
                                new_n = new_n * (z_score / z_alpha) ** 2
                        elif u[0] >= 0 and u[1] < 0:
                            test_stats = u[0] + u[1] - z_score * np.sqrt(2 * (cov[0][0] + cov[1][1] + cov[0][1] + cov[1][0]) / len(y))
                            if test_stats < 0:
                                z_alpha = (u[0] + u[1]) / np.sqrt(2 * (cov[0][0] + cov[1][1] - cov[0][1] - cov[1][0]) / len(y))
                                new_n = new_n * (z_score / z_alpha) ** 2
                        elif u[0] < 0 and u[1] >= 0:
                            test_stats = -(u[0] + u[1] + z_score * np.sqrt(2 * (cov[0][0] + cov[1][1] + cov[0][1] + cov[1][0]) / len(y)))
                            if test_stats < 0:
                                z_alpha = (-u[0] - u[1]) / np.sqrt(2 * (cov[0][0] + cov[1][1] - cov[0][1] - cov[1][0]) / len(y))
                                new_n = new_n * (z_score / z_alpha) ** 2
                        else:
                            test_stats = -(u[0] - u[1] + z_score * np.sqrt(2 * (cov[0][0] + cov[1][1] - cov[0][1] - cov[1][0]) / len(y)))
                            if test_stats < 0:
                                z_alpha = (-u[0] + u[1]) / np.sqrt(2 * (cov[0][0] + cov[1][1] - cov[0][1] - cov[1][0]) / len(y))
                                new_n = new_n * (z_score / z_alpha) ** 2

                        test_result[n_active + 1] = [test_stats, new_n]

                        if testing_verbose:
                            print("Selecting " + str(n_active + 1) + "th varieble: ")
                            print("Correlations: " + str(np.round(u, 4)))
                            print("Test statistics: " + str(round(test_stats, 4)))
                        
                        if testing_stop:
                            if test_stats < 0:
                                if testing_verbose:
                                    print("Not enough samples!")
                                return alphas, active, coefs.T, test_result
                    else:
                        test_result[n_active + 1] = [0, 0]

                C_ = Cov[C_idx]

                if positive:
                    C = C_
                else:
                    C = np.fabs(C_)
            else:
                C = 0.

        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]

        alpha[0] = C / n_samples
        if alpha[0] <= alpha_min + equality_tolerance:  # early stopping
            if abs(alpha[0] - alpha_min) > equality_tolerance:
                # interpolation factor 0 <= ss < 1
                if n_iter > 0:
                    # In the first iteration, all alphas are zero, the formula
                    # below would make ss a NaN
                    ss = ((prev_alpha[0] - alpha_min) /
                          (prev_alpha[0] - alpha[0]))
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break

        if n_iter >= max_iter or n_active >= n_features:
            break
        if not drop:

            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   0 )                                   #
            #     L  ->  (       )  , where L * w = Xa' x_j          #
            #            ( w   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################

            if positive:
                sign_active[n_active] = np.ones_like(C_)
            else:
                sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx + n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov_not_shortened = Cov
            Cov = Cov[1:]  # remove Cov[0]

            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = \
                    np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                # swap does only work inplace if matrix is fortran
                # contiguous ...
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active],
                                        trans=0, lower=1,
                                        overwrite_b=True,
                                        **SOLVE_TRIANGULAR_ARGS)

            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag

            if diag < 1e-7:
                # The system is becoming too ill-conditioned.
                # We have degenerate vectors in our active set.
                # We'll 'drop for good' the last regressor added.

                # Note: this case is very rare. It is no longer triggered by
                # the test suite. The `equality_tolerance` margin added in 0.16
                # to get early stopping to work consistently on all versions of
                # Python including 32 bit Python under Windows seems to make it
                # very difficult to trigger the 'drop for good' strategy.
                warnings.warn('Regressors in active set degenerate. '
                              'Dropping a regressor, after %i iterations, '
                              'i.e. alpha=%.3e, '
                              'with an active set of %i regressors, and '
                              'the smallest cholesky pivot element being %.3e.'
                              ' Reduce max_iter or increase eps parameters.'
                              % (n_iter, alpha, n_active, diag),
                              ConvergenceWarning)

                # XXX: need to figure a 'drop for good' way
                Cov = Cov_not_shortened
                Cov[0] = 0
                Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
                continue

            active.append(indices[n_active])
            n_active += 1

            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, active[-1], '',
                                                      n_active, C))

        if method == 'lasso' and n_iter > 0 and prev_alpha[0] < alpha[0]:
            # alpha is increasing. This is because the updates of Cov are
            # bringing in too much numerical error that is greater than
            # than the remaining correlation with the
            # regressors. Time to bail out
            warnings.warn('Early stopping the lars path, as the residues '
                          'are small and the current value of alpha is no '
                          'longer well controlled. %i iterations, alpha=%.3e, '
                          'previous alpha=%.3e, with an active set of %i '
                          'regressors.'
                          % (n_iter, alpha, prev_alpha, n_active),
                          ConvergenceWarning)
            break

        # least squares solution
        least_squares, _ = solve_cholesky(L[:n_active, :n_active],
                                          sign_active[:n_active],
                                          lower=True)

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, _ = solve_cholesky(
                        L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny32))
        if positive:
            gamma_ = min(g1, C / AA)
        else:
            g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny32))
            gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            # some coefficients have changed sign
            idx = np.where(z == z_pos)[0][::-1]

            # update the sign, important for LAR
            sign_active[idx] = -sign_active[idx]

            if method == 'lasso':
                gamma_ = z_pos
            drop = True

        n_iter += 1

        if return_path:
            if n_iter >= coefs.shape[0]:
                del coef, alpha, prev_alpha, prev_coef
                # resize the coefs and alphas array
                add_features = 2 * max(1, (max_features - n_active))
                coefs = np.resize(coefs, (n_iter + add_features, n_features))
                coefs[-add_features:] = 0
                alphas = np.resize(alphas, n_iter + add_features)
                alphas[-add_features:] = 0
            coef = coefs[n_iter]
            prev_coef = coefs[n_iter - 1]
        else:
            # mimic the effect of incrementing n_iter on the array references
            prev_coef = coef
            prev_alpha[0] = alpha[0]
            coef = np.zeros_like(coef)

        coef[active] = prev_coef[active] + gamma_ * least_squares

        # update correlations
        Cov -= gamma_ * corr_eq_dir

        # See if any coefficient has changed sign
        if drop and method == 'lasso':

            # handle the case when idx is not length of 1
            for ii in idx:
                arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii)

            n_active -= 1
            # handle the case when idx is not length of 1
            drop_idx = [active.pop(ii) for ii in idx]

            if Gram is None:
                # propagate dropped variable
                for ii in idx:
                    for i in range(ii, n_active):
                        X.T[i], X.T[i + 1] = swap(X.T[i], X.T[i + 1])
                        # yeah this is stupid
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]

                # TODO: this could be updated
                residual = y - np.dot(X[:, :n_active], coef[active])
                temp = np.dot(X.T[n_active], residual)

                Cov = np.r_[temp, Cov]
            else:
                for ii in idx:
                    for i in range(ii, n_active):
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]
                        Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i + 1])
                        Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i],
                                                          Gram[:, i + 1])

                # Cov_n = Cov_j + x_j * X + increment(betas) TODO:
                # will this still work with multiple drops ?

                # recompute covariance. Probably could be done better
                # wrong as Xy is not swapped with the rest of variables

                # TODO: this could be updated
                temp = Cov_copy[drop_idx] - np.dot(Gram_copy[drop_idx], coef)
                Cov = np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.)  # just to maintain size
            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, '', drop_idx,
                                                      n_active, abs(temp)))

    if return_path:
        # resize coefs in case of early stop
        alphas = alphas[:n_iter + 1]
        coefs = coefs[:n_iter + 1]

        if return_n_iter:
            return alphas, active, coefs.T, n_iter
        else:
            if testing:
                return alphas, active, coefs.T, test_result
            else:
                return alphas, active, coefs.T
    else:
        if return_n_iter:
            return alpha, active, coef, n_iter
        else:
            return alpha, active, coef