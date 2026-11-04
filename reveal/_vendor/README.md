# Bundled third-party code

REVEAL ships two packages from the mdml repository (github.com/svats73/mdml, "mdml-main"),
so that `pip install` gives everything needed. They are not modified apart from six import lines.

| folder | origin | license | used by |
|---|---|---|---|
| `timelaggedcv/` | `mdml/timelaggedcv` | mdml repository license, GNU LGPL v2.1 (`timelaggedcv/LICENSE`; the folder has no license of its own) | `reveal reduce tae` |
| `sksfa/` | `mdml/sklearn-sfa/sksfa`: sklearn-sfa 0.1.6 with mdml's patch adding `fit(X, tau)` (differences `X[tau:] - X[:-tau]`) | sklearn-sfa BSD-style license, © 2020 Merlin Schüler and sklearn-sfa contributors (`sksfa/LICENSE`) | `reveal reduce sfa` |

Changes (absolute self-imports made relative, so the packages run from inside `reveal._vendor`):

    timelaggedcv/model.py            from timelaggedcv.dataset import ...  ->  from .dataset import ...
    timelaggedcv/estimator.py        from timelaggedcv.dataset import ...  ->  from .dataset import ...
    timelaggedcv/net.py              from timelaggedcv.vamp import ...     ->  from .vamp import ...
    timelaggedcv/dataset/dataset.py  from timelaggedcv import ...          ->  from .. import ...
    sksfa/_hsfa.py                   from sksfa.utils import ...           ->  from .utils import ...
    sksfa/_hsfa.py                   from sksfa import SFA                 ->  from ._sfa import SFA

sklearn-sfa's own test folder is not included.
Because they live under `reveal._vendor`, they never clash with a separately installed sklearn-sfa
or mdml; REVEAL always uses these copies (except `reveal reduce tae --tlcv-path DIR`, which imports
`timelaggedcv` from DIR instead).
