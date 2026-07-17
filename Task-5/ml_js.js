// =====================================
//  纯 JavaScript 机器学习分类器库
//  接口：.fit(X, y) .predict(X) .predictProba(X)
// =====================================

// ----- 工具函数 -----
function accuracy(y1, y2) {
  var c = 0;
  for (var i = 0; i < y1.length; i++) if (y1[i] === y2[i]) c++;
  return c / y1.length;
}
function precision(y1, y2) {
  var tp = 0, fp = 0;
  for (var i = 0; i < y1.length; i++) {
    if (y2[i] === 1) { if (y1[i] === 1) tp++; else fp++; }
  }
  return tp + fp > 0 ? tp / (tp + fp) : 0;
}
function recall(y1, y2) {
  var tp = 0, fn = 0;
  for (var i = 0; i < y1.length; i++) {
    if (y1[i] === 1) { if (y2[i] === 1) tp++; else fn++; }
  }
  return tp + fn > 0 ? tp / (tp + fn) : 0;
}
function f1Score(y1, y2) {
  var p = precision(y1, y2), r = recall(y1, y2);
  return p + r > 0 ? 2 * p * r / (p + r) : 0;
}
function confusionMat(y1, y2) {
  var tp = 0, tn = 0, fp = 0, fn = 0;
  for (var i = 0; i < y1.length; i++) {
    if (y2[i] === 1 && y1[i] === 1) tp++;
    else if (y2[i] === 0 && y1[i] === 0) tn++;
    else if (y2[i] === 1 && y1[i] === 0) fn++;
    else fp++;
  }
  return { tp: tp, tn: tn, fp: fp, fn: fn };
}
function rocCurve(y_true, y_prob) {
  var pts = y_true.map(function(v, i) { return { prob: y_prob[i], label: v }; });
  pts.sort(function(a, b) { return b.prob - a.prob; });
  var tpr = [0], fpr = [0];
  var pos = y_true.filter(function(v) { return v === 1; }).length;
  var neg = y_true.length - pos;
  var tp = 0, fp = 0;
  for (var i = 0; i < pts.length; i++) {
    if (pts[i].label === 1) tp++; else fp++;
    tpr.push(pos > 0 ? tp / pos : 0);
    fpr.push(neg > 0 ? fp / neg : 0);
  }
  return { fpr: fpr, tpr: tpr };
}
// 排列重要性：打乱单列特征，观察准确率下降
function permutationImportance(model, X, y, keys, nRepeats) {
  nRepeats = nRepeats || 3;
  var baseline = accuracy(y, model.predict(X));
  var imp = [];
  for (var j = 0; j < keys.length; j++) {
    var drop = 0;
    for (var r = 0; r < nRepeats; r++) {
      var Xp = X.map(function(row) { return row.slice(); });
      var idx = [];
      for (var i = 0; i < X.length; i++) idx.push(i);
      // Fisher-Yates shuffle on column j
      for (var k = X.length - 1; k > 0; k--) {
        var ri = Math.floor(Math.random() * (k + 1));
        var tmp = Xp[k][j];
        Xp[k][j] = Xp[ri][j];
        Xp[ri][j] = tmp;
      }
      drop += baseline - accuracy(y, model.predict(Xp));
    }
    imp.push({ f: keys[j], v: Math.abs(drop / nRepeats) });
  }
  return imp.sort(function(a, b) { return b.v - a.v; }).slice(0, 15);
}
// 别名，供 HTML 引用
var f1 = f1Score;
function rocAuc(y_true, y_prob) {
  var roc = rocCurve(y_true, y_prob);
  var auc = 0;
  for (var i = 1; i < roc.fpr.length; i++)
    auc += (roc.fpr[i] - roc.fpr[i - 1]) * (roc.tpr[i] + roc.tpr[i - 1]) / 2;
  return auc;
}
function std(mat) {
  var n = mat.length, m = mat[0].length;
  var mean = [], stdv = [];
  for (var j = 0; j < m; j++) {
    var s = 0;
    for (var i = 0; i < n; i++) s += mat[i][j];
    mean[j] = s / n;
  }
  for (j = 0; j < m; j++) {
    var ss = 0;
    for (i = 0; i < n; i++) ss += (mat[i][j] - mean[j]) * (mat[i][j] - mean[j]);
    stdv[j] = Math.sqrt(ss / n) + 1e-10;
  }
  var out = [];
  for (i = 0; i < n; i++) {
    out[i] = [];
    for (j = 0; j < m; j++) out[i][j] = (mat[i][j] - mean[j]) / stdv[j];
  }
  return out;
}
function matMul(a, b) {
  var m = a.length, n = b[0].length, p = b.length;
  var out = [];
  for (var i = 0; i < m; i++) {
    out[i] = [];
    for (var j = 0; j < n; j++) {
      var s = 0;
      for (var k = 0; k < p; k++) s += a[i][k] * b[k][j];
      out[i][j] = s;
    }
  }
  return out;
}
function matT(a) {
  var m = a.length, n = a[0].length;
  var out = [];
  for (var j = 0; j < n; j++) { out[j] = []; for (var i = 0; i < m; i++) out[j][i] = a[i][j]; }
  return out;
}
function sigmoid(z) { return 1 / (1 + Math.exp(-Math.max(-100, Math.min(100, z)))); }

// =====================================
//  1. LogisticRegressionJS
// =====================================
function LogisticRegressionJS(C, classWeight, lr, maxIter) {
  this.C = C || 1.0;
  this.lr = lr || 0.1;
  this.maxIter = maxIter || 300;
  this.w = null;
  this.b = 0;
}
LogisticRegressionJS.prototype.fit = function(X, y) {
  var n = X.length, m = X[0].length;
  this.coef_ = new Array(m);
  for (var j = 0; j < m; j++) this.coef_[j] = 0;
  this.b = 0;
  for (var iter = 0; iter < this.maxIter; iter++) {
    var dw = new Array(m), db = 0;
    for (j = 0; j < m; j++) dw[j] = 0;
    for (var i = 0; i < n; i++) {
      var z = this.b;
      for (j = 0; j < m; j++) z += this.coef_[j] * X[i][j];
      var p = sigmoid(z);
      var err = p - y[i];
      for (j = 0; j < m; j++) dw[j] += err * X[i][j];
      db += err;
    }
    var reg = 0.5 / this.C;
    for (j = 0; j < m; j++) {
      dw[j] = dw[j] / n + reg * this.coef_[j] / n;
      this.coef_[j] -= this.lr * dw[j];
    }
    this.b -= this.lr * db / n;
  }
  this._fitted = true;
};
LogisticRegressionJS.prototype.getCoef = function() { return this.coef_; };
LogisticRegressionJS.prototype.predict = function(X) {
  return this.predictProba(X).map(function(p) { return p >= 0.5 ? 1 : 0; });
};
LogisticRegressionJS.prototype.predictProba = function(X) {
  var out = [];
  for (var i = 0; i < X.length; i++) {
    var z = this.b;
    for (var j = 0; j < X[i].length; j++) z += this.coef_[j] * X[i][j];
    out.push(sigmoid(z));
  }
  return out;
};

// =====================================
//  2. DecisionTreeJS
// =====================================
function DecisionTreeJS(maxDepth, minSamples) {
  this.maxDepth = maxDepth || 5;
  this.minSamples = minSamples || 3;
  this.tree = null;
}
DecisionTreeJS.prototype._gini = function(y) {
  var n = y.length;
  if (n === 0) return 1;
  var c1 = 0;
  for (var i = 0; i < n; i++) if (y[i] === 1) c1++;
  var p = c1 / n;
  return 1 - p * p - (1 - p) * (1 - p);
};
DecisionTreeJS.prototype._bestSplit = function(X, y) {
  var n = X.length, m = X[0].length;
  var bestG = this._gini(y), bestJ = -1, bestV = 0, bestL = [], bestR = [];
  for (var j = 0; j < m; j++) {
    var vals = [];
    for (var i = 0; i < n; i++) vals.push(X[i][j]);
    var uniq = vals.filter(function(v, idx) { return vals.indexOf(v) === idx; }).sort(function(a, b) { return a - b; });
    var step = Math.max(1, Math.floor(uniq.length / 20));
    for (var k = step; k < uniq.length; k += step) {
      var th = uniq[k];
      var lI = [], rI = [];
      for (var i = 0; i < n; i++) {
        if (X[i][j] <= th) lI.push(i); else rI.push(i);
      }
      if (lI.length < this.minSamples || rI.length < this.minSamples) continue;
      var gini = (lI.length / n) * this._gini(lI.map(function(i) { return y[i]; })) +
                 (rI.length / n) * this._gini(rI.map(function(i) { return y[i]; }));
      if (gini < bestG) { bestG = gini; bestJ = j; bestV = th; bestL = lI; bestR = rI; }
    }
  }
  return { gain: bestG, j: bestJ, v: bestV, l: bestL, r: bestR };
};
DecisionTreeJS.prototype._build = function(X, y, depth) {
  var n = y.length;
  var c1 = 0; for (var i = 0; i < n; i++) if (y[i] === 1) c1++;
  if (depth >= this.maxDepth || n <= this.minSamples || c1 === 0 || c1 === n) {
    return { leaf: true, value: c1 >= n - c1 ? 1 : 0, prob: c1 / n };
  }
  var sp = this._bestSplit(X, y);
  if (sp.j === -1) return { leaf: true, value: c1 >= n - c1 ? 1 : 0, prob: c1 / n };
  var lX = sp.l.map(function(i) { return X[i]; }), ly = sp.l.map(function(i) { return y[i]; });
  var rX = sp.r.map(function(i) { return X[i]; }), ry = sp.r.map(function(i) { return y[i]; });
  return {
    leaf: false, j: sp.j, v: sp.v,
    left: this._build(lX, ly, depth + 1),
    right: this._build(rX, ry, depth + 1),
    prob: c1 / n
  };
};
DecisionTreeJS.prototype._predOne = function(x, node) {
  if (node.leaf) return node;
  return x[node.j] <= node.v ? this._predOne(x, node.left) : this._predOne(x, node.right);
};
DecisionTreeJS.prototype.fit = function(X, y) {
  this.tree = this._build(X, y, 0);
};
DecisionTreeJS.prototype.predict = function(X) {
  var self = this;
  return X.map(function(x) { return self._predOne(x, self.tree).value; });
};
DecisionTreeJS.prototype.predictProba = function(X) {
  var self = this;
  return X.map(function(x) { return self._predOne(x, self.tree).prob; });
};

// =====================================
//  3. RandomForestJS
// =====================================
function RandomForestJS(nEstimators, maxDepth) {
  this.nEstimators = nEstimators || 100;
  this.maxDepth = maxDepth || 5;
  this.trees = [];
}
RandomForestJS.prototype._bootstrap = function(X, y) {
  var n = X.length, bX = [], bY = [];
  for (var i = 0; i < n; i++) {
    var idx = Math.floor(Math.random() * n);
    bX.push(X[idx]); bY.push(y[idx]);
  }
  return { X: bX, y: bY };
};
RandomForestJS.prototype.fit = function(X, y) {
  this.trees = [];
  for (var t = 0; t < this.nEstimators; t++) {
    var boot = this._bootstrap(X, y);
    var tree = new DecisionTreeJS(this.maxDepth, 2);
    tree.fit(boot.X, boot.y);
    this.trees.push(tree);
  }
};
RandomForestJS.prototype.predict = function(X) {
  var self = this;
  return X.map(function(x) {
    var votes = 0;
    for (var t = 0; t < self.trees.length; t++) votes += self.trees[t].predict([x])[0];
    return votes >= self.trees.length / 2 ? 1 : 0;
  });
};
RandomForestJS.prototype.predictProba = function(X) {
  var self = this;
  return X.map(function(x) {
    var sum = 0;
    for (var t = 0; t < self.trees.length; t++) sum += self.trees[t].predict([x])[0];
    return sum / self.trees.length;
  });
};

// =====================================
//  3b. RegressionTreeJS (for GradientBoosting)
// =====================================
function RegressionTreeJS(maxDepth, minSamples) {
  this.maxDepth = maxDepth || 3;
  this.minSamples = minSamples || 5;
  this.tree = null;
}
RegressionTreeJS.prototype._variance = function(y) {
  var n = y.length;
  if (n <= 1) return 0;
  var mean = y.reduce(function(a,v){return a+v;},0) / n;
  var varSum = y.reduce(function(a,v){return a+(v-mean)*(v-mean);},0);
  return varSum / n;
};
RegressionTreeJS.prototype._bestSplit = function(X, y) {
  var n = X.length, m = X[0].length;
  var parentVar = this._variance(y);
  var bestG = parentVar, bestJ = -1, bestV = 0, bestL = [], bestR = [];
  for (var j = 0; j < m; j++) {
    var vals = [];
    for (var i = 0; i < n; i++) vals.push(X[i][j]);
    var uniq = vals.filter(function(v, idx) { return vals.indexOf(v) === idx; }).sort(function(a, b) { return a - b; });
    // 限制最多尝试 20 个分位点，提速 10x
    var step = Math.max(1, Math.floor(uniq.length / 20));
    for (var k = step; k < uniq.length; k += step) {
      var th = uniq[k];
      var lI = [], rI = [];
      for (var i = 0; i < n; i++) {
        if (X[i][j] <= th) lI.push(i); else rI.push(i);
      }
      if (lI.length < this.minSamples || rI.length < this.minSamples) continue;
      var varRed = (lI.length / n) * this._variance(lI.map(function(i) { return y[i]; })) +
                   (rI.length / n) * this._variance(rI.map(function(i) { return y[i]; }));
      if (varRed < bestG) { bestG = varRed; bestJ = j; bestV = th; bestL = lI; bestR = rI; }
    }
  }
  return { gain: bestG, j: bestJ, v: bestV, l: bestL, r: bestR };
};
RegressionTreeJS.prototype._build = function(X, y, depth) {
  var n = y.length;
  var mean = y.reduce(function(a,v){return a+v;},0) / n;
  if (depth >= this.maxDepth || n <= this.minSamples) {
    return { leaf: true, value: mean };
  }
  var sp = this._bestSplit(X, y);
  if (sp.j === -1) return { leaf: true, value: mean };
  var lX = sp.l.map(function(i) { return X[i]; }), ly = sp.l.map(function(i) { return y[i]; });
  var rX = sp.r.map(function(i) { return X[i]; }), ry = sp.r.map(function(i) { return y[i]; });
  return {
    leaf: false, j: sp.j, v: sp.v, value: mean,
    left: this._build(lX, ly, depth + 1),
    right: this._build(rX, ry, depth + 1)
  };
};
RegressionTreeJS.prototype._predOne = function(x, node) {
  if (node.leaf) return node.value;
  return x[node.j] <= node.v ? this._predOne(x, node.left) : this._predOne(x, node.right);
};
RegressionTreeJS.prototype.fit = function(X, y) {
  this.tree = this._build(X, y, 0);
};
RegressionTreeJS.prototype.predict = function(X) {
  var self = this;
  return X.map(function(x) { return self._predOne(x, self.tree); });
};
RegressionTreeJS.prototype.predictProba = RegressionTreeJS.prototype.predict;

// =====================================
//  4. GradientBoostingJS
// =====================================
function GradientBoostingJS(nEstimators, lr, maxDepth) {
  this.nEstimators = nEstimators || 100;
  this.lr = lr || 0.1;
  this.maxDepth = maxDepth || 3;
  this.trees = [];
  this.basePred = 0;
}
GradientBoostingJS.prototype.fit = function(X, y) {
  var n = X.length;
  var c1 = 0; for (var i = 0; i < n; i++) c1 += y[i];
  this.basePred = Math.log((c1 + 1) / (n - c1 + 1));
  var raw = new Array(n);
  for (i = 0; i < n; i++) raw[i] = this.basePred;
  this.trees = [];
  for (var t = 0; t < this.nEstimators; t++) {
    var resid = [];
    for (i = 0; i < n; i++) {
      var p = 1 / (1 + Math.exp(-raw[i]));
      resid.push(y[i] - p);
    }
    var tree = new RegressionTreeJS(Math.min(this.maxDepth, 3), 5);
    tree.fit(X, resid);
    this.trees.push(tree);
    for (i = 0; i < n; i++) raw[i] += this.lr * tree.predict([X[i]])[0];
  }
};
GradientBoostingJS.prototype._rawScore = function(x) {
  var s = this.basePred;
  for (var t = 0; t < this.trees.length; t++)
    s += this.lr * this.trees[t].predict([x])[0];
  return s;
};
GradientBoostingJS.prototype.predict = function(X) {
  var self = this;
  return X.map(function(x) { return sigmoid(self._rawScore(x)) >= 0.5 ? 1 : 0; });
};
GradientBoostingJS.prototype.predictProba = function(X) {
  var self = this;
  return X.map(function(x) { return sigmoid(self._rawScore(x)); });
};

// =====================================
//  5. LinearSVCJS (hinge loss SGD)
// =====================================
function LinearSVCJS(C, lr, maxIter) {
  this.C = C || 1.0;
  this.lr = lr || 0.01;
  this.maxIter = maxIter || 200;
  this.w = null;
  this.b = 0;
}
LinearSVCJS.prototype.fit = function(X, y) {
  var n = X.length, m = X[0].length;
  this.w = new Array(m);
  for (var j = 0; j < m; j++) this.w[j] = 0;
  this.b = 0;
  var y2 = y.map(function(v) { return v === 0 ? -1 : 1; });
  for (var iter = 0; iter < this.maxIter; iter++) {
    for (var i = 0; i < n; i++) {
      var score = this.b;
      for (j = 0; j < m; j++) score += this.w[j] * X[i][j];
      if (y2[i] * score < 1) {
        var eta = this.lr / (1 + iter * 0.001);
        for (j = 0; j < m; j++) this.w[j] = (1 - eta) * this.w[j] + eta * this.C * y2[i] * X[i][j];
        this.b += eta * this.C * y2[i];
      } else {
        var eta2 = this.lr / (1 + iter * 0.001);
        for (j = 0; j < m; j++) this.w[j] = (1 - eta2) * this.w[j];
      }
    }
  }
};
LinearSVCJS.prototype.getCoef = function() { return this.w; };
LinearSVCJS.prototype._decision = function(x) {
  var s = this.b;
  for (var j = 0; j < x.length; j++) s += this.w[j] * x[j];
  return s;
};
LinearSVCJS.prototype.predict = function(X) {
  var self = this;
  return X.map(function(x) { return self._decision(x) >= 0 ? 1 : 0; });
};
LinearSVCJS.prototype.predictProba = function(X) {
  var self = this;
  return X.map(function(x) { return sigmoid(self._decision(x)); });
};
