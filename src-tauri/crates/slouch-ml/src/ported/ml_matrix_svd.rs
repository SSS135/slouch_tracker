//! Right singular vectors from the Golub-Reinsch/JAMA algorithm used by
//! `ml-matrix` 6.x. The operation order intentionally mirrors JavaScript.

pub(crate) fn right_singular_vectors(input: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let original_rows = input.len();
    let original_columns = input[0].len();
    let swapped = original_rows < original_columns;
    let mut a = if swapped {
        transpose(input)
    } else {
        input.to_vec()
    };
    let m = a.len();
    let n = a[0].len();
    let want_u = swapped;
    let want_v = !swapped;
    let nu = m.min(n);
    let ni = (m + 1).min(n);
    let mut s = vec![0.0_f64; ni];
    let mut u = vec![vec![0.0_f64; nu]; m];
    let mut v = vec![vec![0.0_f64; n]; n];
    let mut e = vec![0.0_f64; n];
    let mut work = vec![0.0_f64; m];
    let nct = (m - 1).min(n);
    let nrt = n.saturating_sub(2).min(m);
    for k in 0..nct.max(nrt) {
        if k < nct {
            for row in a.iter().skip(k) {
                s[k] = hypotenuse(s[k], row[k]);
            }
            if s[k] != 0.0 {
                if a[k][k] < 0.0 {
                    s[k] = -s[k];
                }
                for row in a.iter_mut().skip(k) {
                    row[k] /= s[k];
                }
                a[k][k] += 1.0;
            }
            s[k] = -s[k];
        }
        for j in k + 1..n {
            if k < nct && s[k] != 0.0 {
                let mut t = 0.0;
                for row in a.iter().skip(k) {
                    t += row[k] * row[j];
                }
                t = -t / a[k][k];
                for row in a.iter_mut().skip(k) {
                    row[j] += t * row[k];
                }
            }
            e[j] = a[k][j];
        }
        if want_u && k < nct {
            for i in k..m {
                u[i][k] = a[i][k];
            }
        }
        if k < nrt {
            for index in k + 1..n {
                e[k] = hypotenuse(e[k], e[index]);
            }
            if e[k] != 0.0 {
                if e[k + 1] < 0.0 {
                    e[k] = -e[k];
                }
                let divisor = e[k];
                for value in e.iter_mut().skip(k + 1) {
                    *value /= divisor;
                }
                e[k + 1] += 1.0;
            }
            e[k] = -e[k];
            if k + 1 < m && e[k] != 0.0 {
                for value in work.iter_mut().skip(k + 1) {
                    *value = 0.0;
                }
                for i in k + 1..m {
                    for j in k + 1..n {
                        work[i] += e[j] * a[i][j];
                    }
                }
                for j in k + 1..n {
                    let t = -e[j] / e[k + 1];
                    for i in k + 1..m {
                        a[i][j] += t * work[i];
                    }
                }
            }
            if want_v {
                for i in k + 1..n {
                    v[i][k] = e[i];
                }
            }
        }
    }
    let mut p = n.min(m + 1);
    if nct < n {
        s[nct] = a[nct][nct];
    }
    if m < p {
        s[p - 1] = 0.0;
    }
    if nrt + 1 < p {
        e[nrt] = a[nrt][p - 1];
    }
    e[p - 1] = 0.0;
    if want_u {
        for (j, row) in u.iter_mut().enumerate().take(nu).skip(nct) {
            row[j] = 1.0;
        }
        for k in (0..nct).rev() {
            if s[k] != 0.0 {
                for j in k + 1..nu {
                    let mut t = 0.0;
                    for row in u.iter().skip(k) {
                        t += row[k] * row[j];
                    }
                    t = -t / u[k][k];
                    for row in u.iter_mut().skip(k) {
                        row[j] += t * row[k];
                    }
                }
                for row in u.iter_mut().skip(k) {
                    row[k] = -row[k];
                }
                u[k][k] += 1.0;
                for row in u.iter_mut().take(k.saturating_sub(1)) {
                    row[k] = 0.0;
                }
            } else {
                for row in &mut u {
                    row[k] = 0.0;
                }
                u[k][k] = 1.0;
            }
        }
    }
    if want_v {
        for k in (0..n).rev() {
            if k < nrt && e[k] != 0.0 {
                for j in k + 1..n {
                    let mut t = 0.0;
                    for row in v.iter().skip(k + 1) {
                        t += row[k] * row[j];
                    }
                    t = -t / v[k + 1][k];
                    for row in v.iter_mut().skip(k + 1) {
                        row[j] += t * row[k];
                    }
                }
            }
            for row in &mut v {
                row[k] = 0.0;
            }
            v[k][k] = 1.0;
        }
    }
    let pp = p - 1;
    while p > 0 {
        let mut k = p as isize - 2;
        while k >= 0 {
            let index = k as usize;
            let alpha = f64::MIN_POSITIVE + f64::EPSILON * (s[index] + s[index + 1].abs()).abs();
            if e[index].abs() <= alpha || e[index].is_nan() {
                e[index] = 0.0;
                break;
            }
            k -= 1;
        }
        let kase;
        if k == p as isize - 2 {
            kase = 4;
        } else {
            let mut ks = p as isize - 1;
            while ks >= k {
                if ks == k {
                    break;
                }
                let index = ks as usize;
                let t = if index != p { e[index].abs() } else { 0.0 }
                    + if index != (k + 1) as usize {
                        e[index - 1].abs()
                    } else {
                        0.0
                    };
                if s[index].abs() <= f64::EPSILON * t {
                    s[index] = 0.0;
                    break;
                }
                ks -= 1;
            }
            if ks == k {
                kase = 3;
            } else if ks == p as isize - 1 {
                kase = 1;
            } else {
                kase = 2;
                k = ks;
            }
        }
        k += 1;
        let k = k as usize;
        match kase {
            1 => {
                let mut f = e[p - 2];
                e[p - 2] = 0.0;
                for j in (k..=p - 2).rev() {
                    let mut t = hypotenuse(s[j], f);
                    let cs = s[j] / t;
                    let sn = f / t;
                    s[j] = t;
                    if j != k {
                        f = -sn * e[j - 1];
                        e[j - 1] *= cs;
                    }
                    if want_v {
                        for row in &mut v {
                            t = cs * row[j] + sn * row[p - 1];
                            row[p - 1] = -sn * row[j] + cs * row[p - 1];
                            row[j] = t;
                        }
                    }
                }
            }
            2 => {
                let mut f = e[k - 1];
                e[k - 1] = 0.0;
                for j in k..p {
                    let mut t = hypotenuse(s[j], f);
                    let cs = s[j] / t;
                    let sn = f / t;
                    s[j] = t;
                    f = -sn * e[j];
                    e[j] *= cs;
                    if want_u {
                        for row in &mut u {
                            t = cs * row[j] + sn * row[k - 1];
                            row[k - 1] = -sn * row[j] + cs * row[k - 1];
                            row[j] = t;
                        }
                    }
                }
            }
            3 => {
                let scale = s[p - 1]
                    .abs()
                    .max(s[p - 2].abs())
                    .max(e[p - 2].abs())
                    .max(s[k].abs())
                    .max(e[k].abs());
                let sp = s[p - 1] / scale;
                let spm1 = s[p - 2] / scale;
                let epm1 = e[p - 2] / scale;
                let sk = s[k] / scale;
                let ek = e[k] / scale;
                let b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
                let c = sp * epm1 * (sp * epm1);
                let mut shift = 0.0;
                if b != 0.0 || c != 0.0 {
                    shift = if b < 0.0 {
                        -(b * b + c).sqrt()
                    } else {
                        (b * b + c).sqrt()
                    };
                    shift = c / (b + shift);
                }
                let mut f = (sk + sp) * (sk - sp) + shift;
                let mut g = sk * ek;
                for j in k..p - 1 {
                    let mut t = hypotenuse(f, g);
                    if t == 0.0 {
                        t = f64::MIN_POSITIVE;
                    }
                    let mut cs = f / t;
                    let mut sn = g / t;
                    if j != k {
                        e[j - 1] = t;
                    }
                    f = cs * s[j] + sn * e[j];
                    e[j] = cs * e[j] - sn * s[j];
                    g = sn * s[j + 1];
                    s[j + 1] *= cs;
                    if want_v {
                        for row in &mut v {
                            t = cs * row[j] + sn * row[j + 1];
                            row[j + 1] = -sn * row[j] + cs * row[j + 1];
                            row[j] = t;
                        }
                    }
                    t = hypotenuse(f, g);
                    if t == 0.0 {
                        t = f64::MIN_POSITIVE;
                    }
                    cs = f / t;
                    sn = g / t;
                    s[j] = t;
                    f = cs * e[j] + sn * s[j + 1];
                    s[j + 1] = -sn * e[j] + cs * s[j + 1];
                    g = sn * e[j + 1];
                    e[j + 1] *= cs;
                    if want_u && j < m - 1 {
                        for row in &mut u {
                            t = cs * row[j] + sn * row[j + 1];
                            row[j + 1] = -sn * row[j] + cs * row[j + 1];
                            row[j] = t;
                        }
                    }
                }
                e[p - 2] = f;
            }
            4 => {
                if s[k] <= 0.0 {
                    s[k] = if s[k] < 0.0 { -s[k] } else { 0.0 };
                    if want_v {
                        for row in v.iter_mut().take(pp + 1) {
                            row[k] = -row[k];
                        }
                    }
                }
                let mut index = k;
                while index < pp && s[index] < s[index + 1] {
                    s.swap(index, index + 1);
                    if want_v && index < n - 1 {
                        for row in &mut v {
                            row.swap(index, index + 1);
                        }
                    }
                    if want_u && index < m - 1 {
                        for row in &mut u {
                            row.swap(index, index + 1);
                        }
                    }
                    index += 1;
                }
                p -= 1;
            }
            _ => unreachable!(),
        }
    }
    if swapped {
        (s, u)
    } else {
        (s, v)
    }
}

fn hypotenuse(a: f64, b: f64) -> f64 {
    if a.abs() > b.abs() {
        let ratio = b / a;
        a.abs() * (1.0 + ratio * ratio).sqrt()
    } else if b != 0.0 {
        let ratio = a / b;
        b.abs() * (1.0 + ratio * ratio).sqrt()
    } else {
        0.0
    }
}

fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..matrix[0].len())
        .map(|column| matrix.iter().map(|row| row[column]).collect())
        .collect()
}
