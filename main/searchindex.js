Search.setIndex({"docnames": ["binary", "index", "install", "notebooks/issue_1369", "notebooks/nan_grad", "notebooks/nan_operators", "notebooks/overview", "notebooks/safe_softmax", "notebooks/sparse", "reductions", "unary", "view_and_select"], "filenames": ["binary.rst", "index.rst", "install.rst", "notebooks/issue_1369.ipynb", "notebooks/nan_grad.ipynb", "notebooks/nan_operators.ipynb", "notebooks/overview.ipynb", "notebooks/safe_softmax.ipynb", "notebooks/sparse.ipynb", "reductions.rst", "unary.rst", "view_and_select.rst"], "titles": ["Binary Operations", "MaskedTensor", "Installation", "Efficiency of writing \u201csparse\u201d semantics for Adagrad", "Distinguishing between 0 and NaN gradient", "Implemented missing torch.nan* operators", "Overview of MaskedTensors", "Safe Softmax", "Sparse semantics", "Reductions", "Unary Operations", "View and select functions"], "terms": {"As": [0, 8, 9], "you": [0, 4, 6, 8, 9], "mai": [0, 1], "have": [0, 3, 6, 8], "seen": 0, "tutori": [0, 8], "maskedtensor": [0, 2, 4, 5, 7, 9, 10, 11], "also": [0, 3, 6, 8], "ha": [0, 3, 5, 6, 7, 8], "implement": [0, 1, 6, 8, 9, 11], "caveat": 0, "mask": [0, 1, 2, 3, 5, 7, 8, 9, 10, 11], "two": [0, 6, 8], "tensor": [0, 1, 2, 3, 4, 5, 6, 7, 9, 11], "must": [0, 3, 6, 8], "match": [0, 3, 6, 8], "els": 0, "an": [0, 1, 3, 5, 6, 7, 8, 9], "error": [0, 3], "rais": 0, "note": [0, 1, 2, 6, 8], "need": [0, 3, 6], "support": [0, 3, 4, 5, 6, 9, 11], "particular": [0, 3, 6, 8], "propos": [0, 3, 5, 9], "semant": [0, 1, 5, 9], "how": [0, 3], "thei": [0, 1, 3, 6, 8], "should": [0, 2, 3, 6], "behav": 0, "instead": [0, 3, 4, 5, 6, 8], "pleas": [0, 1, 6, 7, 8, 9], "open": [0, 6], "issu": [0, 1, 3, 6, 7, 8], "github": [0, 1, 4, 6, 8], "For": [0, 3, 6, 7, 8], "now": [0, 6], "we": [0, 1, 2, 3, 4, 5, 6, 8, 10, 11], "decid": 0, "go": 0, "most": [0, 8], "conserv": 0, "ensur": [0, 2], "user": [0, 1, 3, 5, 6, 8], "know": 0, "exactli": 0, "what": 0, "ar": [0, 3, 4, 6, 8, 10, 11], "being": [0, 4, 7], "intent": 0, "about": [0, 6, 8], "decis": 0, "The": [0, 1, 3, 4, 6, 8, 9, 10], "avail": [0, 10], "inplac": [0, 10], "all": [0, 3, 8, 10], "abov": [0, 3, 6, 8, 10], "except": [0, 6, 10], "alwai": [0, 3], "ani": [0, 1, 3, 6, 8, 9], "featur": [0, 1, 7, 8, 9], "request": [0, 1, 7, 8, 9], "file": [0, 8], "thi": [1, 3, 4, 5, 6, 7, 8, 11], "librari": 1, "part": [1, 3], "pytorch": [1, 2, 3, 6, 7, 8], "project": [1, 8], "current": [1, 3, 8, 9, 11], "classifi": 1, "prototyp": 1, "earli": 1, "stage": 1, "feedback": 1, "test": 1, "encourag": 1, "submit": [1, 8], "encount": 1, "etc": 1, "can": [1, 2, 3, 4, 5, 6, 7, 8, 11], "found": [1, 8], "here": [1, 3, 4, 6, 9], "purpos": 1, "serv": 1, "extens": [1, 3], "torch": [1, 2, 3, 6, 7, 8, 11], "especi": 1, "case": [1, 3, 5, 6, 7, 8], "us": [1, 2, 3, 5, 6, 8], "e": [1, 3, 6, 8, 9, 11], "g": [1, 8, 9], "variabl": 1, "length": 1, "nan": [1, 7], "oper": [1, 3, 6, 9, 11], "differenti": 1, "between": [1, 6], "0": [1, 3, 5, 6, 7, 8, 11], "gradient": [1, 3], "variou": [1, 8], "spars": 1, "applic": 1, "see": [1, 4, 6], "more": [1, 3, 4, 6, 7, 8, 9], "detail": [1, 3, 7], "overview": 1, "distinguish": 1, "safe": 1, "softmax": 1, "effici": [1, 8], "write": 1, "adagrad": 1, "miss": [1, 8], "unari": 1, "binari": [1, 6], "reduct": [1, 5], "view": 1, "select": [1, 4], "function": [1, 3, 6, 7, 8], "To": [2, 8], "via": [2, 6], "follow": [2, 3, 8, 9], "command": 2, "nightli": 2, "requir": [2, 8], "1": [2, 3, 4, 5, 6, 7, 8, 11], "12": [2, 5, 6, 11], "stabl": 2, "version": [2, 3], "work": [2, 3, 4, 8], "verifi": 2, "some": [2, 3, 5], "simpl": 2, "code": 2, "like": [2, 6, 8, 9], "import": [2, 3, 4, 5, 6, 7, 8], "from": [2, 3, 4, 5, 6, 7, 8], "masked_tensor": [2, 3, 4, 5, 6, 7, 8, 11], "data": [2, 3, 6, 7, 8, 10, 11], "2": [2, 3, 4, 6, 8], "3": [2, 3, 4, 6, 7, 8, 11], "4": [2, 3, 4, 5, 6, 8, 11], "5": [2, 3, 4, 5, 6, 8, 11], "6": [2, 3, 4, 6, 8, 11], "dtype": [2, 3, 6, 8, 11], "float": [2, 3, 4, 5, 6, 7, 8, 11], "true": [2, 4, 5, 6, 7, 8, 11], "fals": [2, 4, 5, 6, 7, 8, 11], "mt": [2, 8, 11], "output": [2, 9], "look": [2, 3, 8], "0000": [2, 3, 4, 6, 7, 8, 11], "1369": 3, "discuss": 3, "addit": [3, 5, 6], "line": [3, 8], "were": 3, "introduc": [3, 5], "while": [3, 6, 8], "But": [3, 6], "realli": [3, 6], "doesn": 3, "t": [3, 4, 6, 8], "sparsiti": [3, 8], "compress": [3, 8], "optim": 3, "techniqu": 3, "want": [3, 4, 5, 6], "around": [3, 4], "one": [3, 6, 9], "off": 3, "encod": [3, 5], "behavior": [3, 4, 6], "forc": 3, "awar": 3, "storag": [3, 8], "indic": [3, 8], "valu": [3, 6, 8, 9], "let": [3, 6], "s": [3, 4, 8], "illustr": [3, 8], "In": [3, 6, 8, 9, 11], "ll": [3, 6, 10], "point": [3, 5, 6, 8], "out": [3, 4, 5, 6, 7, 8, 10], "when": [3, 5, 8], "i": [3, 6, 8, 11], "unspecifi": [3, 8], "zero": [3, 4, 6, 8], "just": [3, 8], "compar": 3, "contrast": 3, "equival": 3, "written": 3, "end": 3, "snippet": 3, "repeat": [3, 6], "without": [3, 8], "comment": 3, "show": [3, 4], "breviti": 3, "def": 3, "_make_spars": 3, "grad": [3, 6], "grad_indic": 3, "size": [3, 8], "numel": 3, "return": [3, 5, 6], "empty_lik": 3, "sparse_coo_tensor": [3, 8], "don": [3, 6], "param": 3, "arang": [3, 5, 6, 11], "8": [3, 6, 8], "reshap": [3, 6, 11], "v": [3, 8], "float32": 3, "state_sum": 3, "full_lik": 3, "initi": [3, 8], "state": [3, 8], "sum": [3, 4, 6, 8], "print": [3, 4, 5, 6, 8], "n": [3, 6, 8], "to_dens": [3, 8], "hyperparamet": 3, "ep": 3, "1e": 3, "10": [3, 4, 6], "clr": 3, "7": [3, 4, 6, 11], "5000": [3, 6], "coalesc": 3, "updat": 3, "non": [3, 4], "linear": 3, "so": [3, 8], "uniqu": 3, "_indic": 3, "grad_valu": 3, "_valu": 3, "pow": 3, "same": [3, 8], "both": [3, 6, 8, 11], "dens": 3, "memori": [3, 8], "layout": [3, 8], "sinc": [3, 5, 6], "add_": 3, "take": [3, 8], "care": 3, "make": [3, 4, 6], "std": 3, "even": [3, 8], "though": 3, "clearli": 3, "mean": [3, 5, 6], "re": 3, "onli": [3, 4, 8, 10], "appli": [3, 6, 10, 11], "which": [3, 4, 5, 7, 8], "specifi": [3, 8], "drive": 3, "home": [3, 8], "lot": [3, 8], "pass": [3, 8, 11], "sparse_mask": [3, 8], "9": [3, 5, 6], "16": [3, 6], "25": [3, 5], "where": [3, 8], "veri": [3, 6], "diverg": [3, 7], "technic": 3, "enforc": 3, "certain": [3, 8], "pattern": 3, "defin": [3, 4], "undefin": [3, 6], "If": [3, 4, 6, 8, 9], "still": [3, 8], "includ": [3, 11], "materi": 3, "could": [3, 4, 6, 8], "other": [3, 5, 6, 8, 9], "quit": 3, "brittl": 3, "someon": 3, "argu": 3, "small": 3, "scheme": 3, "caus": 3, "densif": 3, "fine": 3, "until": 3, "new": [3, 11], "csr": 3, "bsr": [3, 8], "block": [3, 8], "separ": 3, "type": 3, "each": [3, 8], "variat": 3, "format": [3, 8], "dodg": 3, "concern": 3, "privat": 3, "method": 3, "std_valu": 3, "sqrt_": 3, "div": [3, 4, 8], "becaus": [3, 6], "well": [3, 6, 8], "alpha": 3, "9027": 3, "9015": 3, "9010": 3, "ve": [3, 11], "been": [3, 8], "conflat": 3, "call": 3, "through": [3, 11], "fix": [3, 4], "exampl": [3, 4, 8, 11], "creat": [3, 8, 9], "entir": [3, 5, 7], "set": [3, 4, 7], "paramet": 3, "avoid": [3, 6], "param2": 3, "state_sum2": 3, "to_spars": [3, 8], "masked_grad": 3, "eventu": 3, "construct": 3, "back": 3, "std2": 3, "regular": [3, 6, 8], "easier": [3, 8], "comparison": [3, 5], "0822": 3, "0620": 3, "0498": 3, "nnz": [3, 8], "sparse_coo": [3, 8], "add": [3, 4], "place": 3, "later": [3, 6], "notic": 3, "access": [3, 4, 6], "intern": [3, 4, 8], "gener": [3, 6, 9], "shorter": 3, "sqrt": 3, "inde": [3, 4, 8], "refer": 3, "path": 3, "addcmul_": 3, "addcdiv_": 3, "vanilla": [3, 4, 8], "8329": 3, "8314": 3, "8306": 3, "minim": 3, "And": 3, "good": [3, 6], "measur": 3, "sure": [3, 4, 6], "numpi": 4, "np": [4, 6], "as_masked_tensor": [4, 6], "One": [4, 7], "run": 4, "inabl": 4, "vs": [4, 8], "actual": 4, "below": [4, 6, 8, 11], "wai": [4, 8, 11], "sever": 4, "differ": [4, 6, 8, 11], "fall": 4, "short": 4, "problem": 4, "result": [4, 6, 7, 11], "underli": [4, 8], "clamp": 4, "its": [4, 8], "deriv": 4, "50": 4, "60": [4, 5, 8], "70": 4, "80": 4, "90": 4, "100": 4, "requires_grad": [4, 6], "y": [4, 5, 6], "exp": 4, "ones_lik": 4, "backward": [4, 6], "5400e": 4, "05": 4, "7379e": 4, "03": 4, "0000e": 4, "00": 4, "grad_fn": 4, "wherebackward0": 4, "none": 4, "tmp": 4, "ipykernel_2351": 4, "3791710618": 4, "py": [4, 8], "userwarn": [4, 8], "attribut": 4, "leaf": 4, "Its": 4, "won": 4, "popul": 4, "dure": [4, 6], "autograd": [4, 6], "field": 4, "retain_grad": 4, "mistak": 4, "com": [4, 6, 8], "pull": [4, 6], "30531": 4, "inform": 4, "trigger": [4, 8], "aten": [4, 8], "src": [4, 8], "core": [4, 8], "tensorbodi": 4, "h": 4, "478": 4, "mx": 4, "my": 4, "provid": [4, 8], "subset": [4, 8], "effect": 4, "chang": 4, "element": [4, 6, 8], "them": [4, 6, 8, 10], "0067": 4, "A": [4, 8, 11], "recent": 4, "incarn": 4, "specif": [4, 6], "http": [4, 6, 8], "randn": [4, 6, 7], "b": 4, "c": 4, "ones": [4, 9], "c1": 4, "bgrad1": 4, "retain_graph": 4, "ma": [4, 6], "isnan": [4, 5], "inf": [4, 7], "loss": 4, "expect": 4, "index": [4, 10], "wa": [5, 8], "close": 5, "inclus": 5, "61474": 5, "altern": [5, 8], "nanmean": [5, 9], "alreadi": [5, 7], "land": 5, "32": 5, "x": [5, 6, 7], "fmod": 5, "masked_fil": [5, 7], "21": [5, 6], "20": [5, 6, 8], "33": 5, "13": [5, 6], "28": 5, "45": 5, "17": [5, 6], "36": 5, "57": 5, "44": 5, "69": 5, "52": 5, "81": 5, "29": 5, "93": 5, "6667": [5, 8], "further": 5, "fulli": [5, 6], "would": [5, 8, 9], "given": [5, 8, 10], "completet": 5, "hand": 5, "input": [5, 6, 8, 9, 10], "empti": 5, "fill_": 5, "first": [6, 8], "m0": 6, "m1": 6, "co": 6, "try": 6, "treat": 6, "shape": [6, 8], "relax": 6, "reason": 6, "revisit": 6, "onc": 6, "valueerror": 6, "5403": 6, "9900": 6, "intersect": 6, "factori": 6, "invert": 6, "similar": [6, 8], "nn": 6, "mha": 6, "d": [6, 8], "logical_and": 6, "get": 6, "store": [6, 8], "logical_or": 6, "suggest": 6, "section": 6, "why": [6, 7], "npm0": 6, "masked_arrai": 6, "npm1": 6, "give": 6, "conveni": 6, "convert": [6, 8], "fill": 6, "cours": 6, "opportun": 6, "altogeth": 6, "check": 6, "whether": [6, 8], "choos": 6, "presum": 6, "expens": 6, "allreduc": 6, "everi": 6, "time": 6, "m0t": 6, "to_tensor": 6, "m1t": 6, "m2t": 6, "2d": 6, "setup": 6, "mul": [6, 8], "int": 6, "randint": 6, "bool": [6, 8], "m": [6, 7, 8], "14": 6, "22": 6, "base": 6, "rfc": [6, 9], "27": 6, "prod": 6, "min": 6, "amin": [6, 8], "max": 6, "amax": 6, "324": 6, "40": [6, 8], "30": [6, 8], "probabl": 6, "allow": 6, "data0": 6, "data1": 6, "mask0": 6, "mask1": 6, "nnpm0": 6, "nnpm1": 6, "11": [6, 11], "18": 6, "19": 6, "38": 6, "15": 6, "23": 6, "34": 6, "42": 6, "46": 6, "associ": 6, "howev": 6, "least": 6, "ask": 6, "obvious": 6, "possibl": 6, "think": [6, 8, 9], "cover": 6, "0792": 6, "9683": 6, "3700": 6, "9161": 6, "8073": 6, "9613": 6, "5290": 6, "9581": 6, "3996": 6, "7627": 6, "4052": 6, "2773": 6, "7082": 6, "5614": 6, "4055": 6, "9276": 6, "9012": 6, "6361": 6, "5489": 6, "0996": 6, "8993": 6, "2633": 6, "0727": 6, "5588": 6, "1068": 6, "0962": 6, "4735": 6, "2337": 6, "1719": 6, "9923": 6, "8241": 6, "9662": 6, "4862": 6, "1371": 6, "manual_se": 6, "custom": 6, "maintain": 6, "scalar": 6, "might": 6, "5084": 6, "7935": 6, "3725": 6, "2078": 6, "5820": 6, "6679": 6, "9655": 6, "multipl": 6, "requires_grad_": 6, "broadcast": 6, "sens": 6, "consid": [6, 11], "complex": 6, "higher": 6, "dimension": [6, 8], "thu": 6, "love": 6, "find": [6, 7, 8, 9], "rigor": 6, "definit": 6, "z": 6, "nx": 6, "commonli": 7, "come": 7, "up": [7, 8], "necess": 7, "batch": 7, "consist": [7, 8], "pad": 7, "translat": 7, "lead": 7, "train": 7, "help": 7, "55056": 7, "luckili": 7, "solv": 7, "5169": 7, "4831": 7, "quickli": 8, "grow": 8, "area": 8, "demand": 8, "due": 8, "comput": 8, "meant": 8, "conjunct": 8, "link": 8, "ultim": 8, "build": 8, "proven": 8, "power": 8, "varieti": 8, "primer": 8, "practition": 8, "major": 8, "equal": 8, "high": 8, "degre": 8, "lower": 8, "advantag": 8, "substructur": 8, "within": 8, "matrix": 8, "There": 8, "number": [8, 11], "leverag": 8, "tradeoff": 8, "adopt": 8, "long": 8, "histori": 8, "formal": 8, "certainli": 8, "partial": 8, "born": 8, "nan_grad": 8, "address": 8, "goal": 8, "becom": 8, "primari": 8, "sourc": 8, "truth": 8, "class": 8, "citizen": 8, "afterthought": 8, "csc": 8, "develop": 8, "futur": 8, "anoth": 8, "stride": 8, "sparse_csr": 8, "accord": 8, "constructor": 8, "recap": 8, "stand": 8, "coordin": 8, "tupl": 8, "correspond": 8, "That": 8, "arrai": 8, "ndim": 8, "nse": 8, "int64": 8, "integ": 8, "do": 8, "either": 8, "sparse_tensor_data": 8, "sparse_tensor_mask": 8, "dense_masked_tensor": 8, "to_sparse_coo": 8, "second": 8, "shown": 8, "nuanc": 8, "behind": 8, "approach": 8, "read": 8, "bottom": 8, "start": 8, "sparse_coo_mt": 8, "similarli": 8, "row": 8, "aim": 8, "decreas": 8, "three": 8, "crow_indic": 8, "entri": 8, "live": 8, "last": 8, "col_indic": 8, "column": 8, "contain": [8, 10], "Of": 8, "beta": 8, "By": [8, 11], "again": 8, "mt_sparse_csr": 8, "to_sparse_csr": 8, "runner": 8, "local": 8, "lib": 8, "python3": 8, "site": 8, "packag": 8, "179": 8, "sparsecsrtensorimpl": 8, "cpp": 8, "66": 8, "sin": 8, "1411": 8, "9589": 8, "v1": 8, "v2": 8, "s1": 8, "s2": 8, "mt1": 8, "mt2": 8, "200": 8, "At": 8, "moment": [8, 10], "across": 8, "dimens": [8, 9], "dim": 8, "next": [8, 9], "mt_spars": 8, "is_spars": 8, "is_sparse_coo": 8, "is_sparse_csr": 8, "recal": 8, "our": 8, "origin": 8, "dense_tensor": 8, "directli": 8, "bring": 8, "word": 8, "warn": 8, "doe": 8, "default": 8, "vast": 8, "under": [8, 11], "hood": [8, 11], "creation": 8, "assum": 8, "thrown": 8, "awai": 8, "therefor": 8, "slightli": 8, "complet": 8, "invalid": 8, "masked_data": 8, "treatment": 8, "mask_valu": 8, "sparse_csr_tensor": 8, "doubl": 8, "float64": 8, "intro": 9, "document": 9, "reduc": 9, "singl": [9, 10], "nansum": 9, "op": [9, 11], "rel": 10, "straightforward": 10, "continu": 10, "wrap": 11, "quick": 11, "list": 11}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"binari": [0, 8], "oper": [0, 5, 8, 10], "maskedtensor": [1, 3, 6, 8], "instal": [1, 2], "tutori": 1, "python": 1, "api": 1, "pip": 2, "verif": 2, "effici": 3, "write": 3, "spars": [3, 8], "semant": [3, 6, 8], "adagrad": 3, "motiv": [3, 7], "code": 3, "origin": 3, "implement": [3, 5], "conclus": 3, "differ": 3, "distinguish": 4, "between": 4, "0": 4, "nan": [4, 5], "gradient": [4, 6], "resolv": 4, "issu": [4, 5], "pytorch": 4, "10729": 4, "torch": [4, 5], "where": 4, "52248": 4, "anoth": 4, "67180": 4, "nansum": 4, "nanmean": 4, "4132": 4, "when": 4, "us": 4, "mask": [4, 6], "x": 4, "yield": 4, "grad": 4, "miss": 5, "21987": 5, "overview": 6, "basic": 6, "vs": 6, "numpi": 6, "s": 6, "maskedarrai": 6, "reduct": [6, 8, 9], "index": 6, "advanc": 6, "exampl": 6, "safe": 7, "softmax": 7, "introduct": 8, "principl": 8, "coo": 8, "tensor": 8, "csr": 8, "support": 8, "unari": [8, 10], "method": 8, "appendix": 8, "construct": 8, "view": 11, "select": 11, "function": 11}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})