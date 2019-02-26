#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define REP(i, n) for (int i = 0; (i) < (int)(n); ++ (i))
#define REP3(i, m, n) for (int i = (m); (i) < (int)(n); ++ (i))
#define REP_R(i, n) for (int i = int(n) - 1; (i) >= 0; -- (i))
#define REP3R(i, m, n) for (int i = int(n) - 1; (i) >= (int)(m); -- (i))
#define ALL(x) begin(x), end(x)
using ll = long long;
using namespace std;
template <class T> using reversed_priority_queue = priority_queue<T, vector<T>, greater<T> >;
template <class T, class U> inline void chmax(T & a, U const & b) { a = max<T>(a, b); }
template <class T, class U> inline void chmin(T & a, U const & b) { a = min<T>(a, b); }
template <typename X, typename T> auto vectors(X x, T a) { return vector<T>(x, a); }
template <typename X, typename Y, typename Z, typename... Zs> auto vectors(X x, Y y, Z z, Zs... zs) { auto cont = vectors(y, z, zs...); return vector<decltype(cont)>(x, cont); }
template <typename T> ostream & operator << (ostream & out, vector<T> const & xs) { REP (i, int(xs.size()) - 1) out << xs[i] << ' '; if (not xs.empty()) out << xs.back(); return out; }
template <typename T> T gcd(T a, T b) { while (a) { b %= a; swap(a, b); } return b; }
template <typename T> T lcm(T a, T b) { return a / gcd(a, b) * b; }

class xor_shift_128 {
public:
    typedef uint32_t result_type;
    xor_shift_128(uint32_t seed = 42) {
        set_seed(seed);
    }
    void set_seed(uint32_t seed) {
        a = seed = 1812433253u * (seed ^ (seed >> 30));
        b = seed = 1812433253u * (seed ^ (seed >> 30)) + 1;
        c = seed = 1812433253u * (seed ^ (seed >> 30)) + 2;
        d = seed = 1812433253u * (seed ^ (seed >> 30)) + 3;
    }
    uint32_t operator() () {
        uint32_t t = (a ^ (a << 11));
        a = b; b = c; c = d;
        return d = (d ^ (d >> 19)) ^ (t ^ (t >> 8));
    }
    static constexpr uint32_t max() { return numeric_limits<result_type>::max(); }
    static constexpr uint32_t min() { return numeric_limits<result_type>::min(); }
private:
    uint32_t a, b, c, d;
};

struct term_t {
    int c;
    vector<int> v;
};
term_t make_term(int c, vector<int> const & v) {
    return (term_t) { c, v };
}

int get_maxcoeff(vector<term_t> const & f) {
    int max_c = 0;
    for (auto const & t : f) {
        if (t.v.empty()) continue;  // ignore constants
        chmax(max_c, t.c);
    }
    return max_c;
}

int apply_all_true(vector<term_t> const & f) {
    int sum_c = 0;
    for (auto const & t : f) {
        sum_c += t.c;
    }
    return sum_c;
}

template <class Generator>
vector<bool> generate_random_vector(int n, Generator & gen) {
    vector<bool> x(n);
    REP (i, n) {
        x[i] = bernoulli_distribution(0.5)(gen);
    }
    return x;
}

int apply_vector(vector<term_t> const & f, vector<bool> const & x) {
    int value = 0;
    for (auto const & t : f) {
        for (int i : t.v) {
            if (not x[i]) {
                goto next;
            }
        }
        value += t.c;
next: ;
    }
    return value;
}

int apply_argumented_vector(vector<term_t> const & g, vector<bool> const & x, vector<bool> const & w) {
    int n = x.size();
    int value = 0;
    for (auto const & t : g) {
        for (int i : t.v) {
            if (not (i < n ? x[i] : w[i - n])) {
                goto next;
            }
        }
        value += t.c;
next: ;
    }
    return value;
}

template <class Generator>
int apply_vector_min_sa(int m, vector<term_t> const & g, vector<bool> const & x, Generator & gen) {
    if (m == 0) return apply_vector(g, x);

    vector<bool> w = generate_random_vector(m, gen);
    int value = apply_argumented_vector(g, x, w);
    int min_value = value;

    int total = max(100, 10 * m);
    REP (iteration, total) {
        double temperature = (double) (total - iteration) / total;

        int i = uniform_int_distribution<int>(0, m - 1)(gen);
        w[i] = not w[i];
        int delta = apply_argumented_vector(g, x, w) - value;

        constexpr double boltzmann = 1;
        if (delta <= 0 or bernoulli_distribution(exp(- boltzmann * delta / temperature))(gen)) {
            value += delta;
            if (value < min_value) {
                min_value = value;
                // cerr << "[*] g(X,W) = " << min_value << " when W = (" << w << ")" << endl;
            }
        } else {
            w[i] = not w[i];
        }
    }
    return min_value;
}

template <class Generator>
int apply_all_true_min_sa(int n, int m, vector<term_t> const & g, Generator & gen) {
    vector<bool> x(n, true);
    return apply_vector_min_sa(m, g, x, gen);
}

double evaluate_score_px(int delta) {
    constexpr int e = 10000;
    constexpr int t = 100;
    if (delta < 0) return - INFINITY;
    return e * (1 - min<double>(t, delta) / t);
}
double evaluate_score_py(int m, int l) {
    constexpr int b = 5;
    return 1000 / (b * m + l + 1000.0);
}
double evaluate_score_pz(int maxcoeff) {
    return 1000 / (maxcoeff + 1000.0);
}

struct quadratic_pseudo_boolean_function {  // as a trait
    virtual ~quadratic_pseudo_boolean_function() = default;
    virtual int new_variable() = 0;
    virtual void use0(int c) = 0;
    virtual void use1(int c, int y1) = 0;
    virtual void use2(int c, int y1, int y2) = 0;
    virtual int get_newvars() const = 0;
    virtual int get_terms() const = 0;
    virtual int get_maxcoeff() const = 0;
    virtual double get_score_py() const {
        return evaluate_score_py(get_newvars(), get_terms());
    }
    virtual double get_score_pz() const {
        return evaluate_score_pz(get_maxcoeff());
    }
    virtual int get_score() const {
        constexpr int a = 10000;
        double px = evaluate_score_px(0);
        double py = get_score_py();
        double pz = get_score_pz();
        return a * px * py * pz;
    }

    void use_term(term_t const & t) {
        if (t.v.size() == 0) {
            use0(t.c);
        } else if (t.v.size() == 1) {
            use1(t.c, t.v[0]);
        } else if (t.v.size() == 2) {
            use2(t.c, t.v[0], t.v[1]);
        } else {
            assert (false);
        }
    }

    void reduce_negative_monomial(term_t const & t) {
        assert (t.c < 0);
        int d = t.v.size();
        int w1 = new_variable();
        use1(- t.c * (d - 1), w1);
        for (int x1 : t.v) {
            use2(t.c, w1, x1);
        }
    }

    void simply_reduce_positive_monomial(term_t const & t) {
        // c
        use0(t.c);
        // - c (1 - x1)
        use0(- t.c);
        use1(t.c, t.v[0]);
        // - c x1 (1 - x2)
        use1(- t.c, t.v[0]);
        use2(t.c, t.v[0], t.v[1]);
        REP3 (i, 2, t.v.size()) {
            // -c x1 x2 .. x{i - 1} (1 - xi)
            int w_i = new_variable();
            use1(t.c * i, w_i);
            use1(- t.c, w_i);
            use2(t.c, w_i, t.v[i]);
            REP (j, i) {
                use2(- t.c, w_i, t.v[j]);
            }
        }
    }

    void reduce_higher_order_clique(term_t const & t) {
        // let S_1 = \sum x_i
        // let S_2 = \sum_i \sum_{i \lt j} x_i x_j
        int d = t.v.size();
        int n_d = (d - 1) / 2;

        REP (i, n_d) {
            int w_i = new_variable();
            if (d % 2 == 1 and i == n_d - 1) {
                // - w_i S_1
                for (int x_i : t.v) {
                    use2(- t.c, w_i, x_i);
                }
                // w_i (2 i + 1)
                use1(t.c * (2 * i + 1), w_i);
            } else {
                // - 2 w_i S_1
                for (int x_i : t.v) {
                    use2(- 2 * t.c, w_i, x_i);
                }
                // w_i (4 i + 3)
                use1(t.c * (4 * i + 3), w_i);
            }
        }
        // S_2
        REP (i, d) {
            REP (j, i) {
                use2(t.c, t.v[i], t.v[j]);
            }
        }
    }

    term_t split_common_parts(term_t t, int x1, int w1) {
        assert (t.c > 0 and t.v.size() >= 3);
        auto it = find(ALL(t.v), x1);
        assert (it != t.v.end());
        use2(t.c, x1, w1);
        *it = w1;
        t.c *= -1;
        reduce_negative_monomial(t);
        t.c *= -1;
        swap(*it, t.v.back());
        t.v.pop_back();
        return t;
    }
};

struct quadratic_pseudo_boolean_function_term_list : public quadratic_pseudo_boolean_function {
    int n;
    int m;
    int c0;
    vector<pair<int, int> > c1;  // [(y1, c)]
    vector<tuple<int, int, int> > c2;  // [(y1, y2, c)]
    quadratic_pseudo_boolean_function_term_list(int n_) {
        n = n_;
        m = 0;
        c0 = 0;
    }

    int new_variable() {
        return n + (m ++);
    }

    void use0(int c) {
        assert (c != 0);
        c0 += c;
    }
    void use1(int c, int y1) {
        assert (c != 0);
        c1.emplace_back(y1, c);
    }
    void use2(int c, int y1, int y2) {
        assert (c != 0);
        if (y1 > y2) swap(y1, y2);
        c2.emplace_back(y1, y2, c);
    }

    void normalize() {
        // c1
        vector<pair<int, int> > c1_;
        sort(ALL(c1));
        for (auto it : c1) {
            if (not c1_.empty() and c1_.back().first == it.first) {
                c1_.back().second += it.second;
                if (c1_.back().second == 0) {
                    c1_.pop_back();
                }
            } else {
                c1_.push_back(it);
            }
        }
        c1.swap(c1_);

        // c2
        vector<tuple<int, int, int> > c2_;
        sort(ALL(c2));
        for (auto it : c2) {
            if (not c2_.empty() and get<0>(c2_.back()) == get<0>(it) and get<1>(c2_.back()) == get<1>(it)) {
                get<2>(c2_.back()) += get<2>(it);
                if (get<2>(c2_.back()) == 0) {
                    c2_.pop_back();
                }
            } else {
                c2_.push_back(it);
            }
        }
        c2.swap(c2_);

        // remove unused variables
        vector<int> used(n + m, -1);
        REP (i, n) {
            used[i] = i;
        }
        int newvars = 0;
        for (auto it : c1) {
            if (used[it.first] == -1) {
                used[it.first] = n + (newvars ++);
            }
        }
        for (auto it : c2) {
            if (used[get<0>(it)] == -1) {
                used[get<0>(it)] = n + (newvars ++);
            }
            if (used[get<1>(it)] == -1) {
                used[get<1>(it)] = n + (newvars ++);
            }
        }
        for (auto & it : c1) {
            it.first = used[it.first];
        }
        for (auto & it : c2) {
            get<0>(it) = used[get<0>(it)];
            get<1>(it) = used[get<1>(it)];
        }
        m = newvars;

        // sort
        for (auto & it : c2) {
            if (get<0>(it) > get<1>(it)) {
                swap(get<0>(it), get<1>(it));
            }
        }
    }

    int get_newvars() const {
        return m;
    }
    int get_terms() const {
        return (c0 != 0) + c1.size() + c2.size();
    }
    int get_maxcoeff() const {
        int maxcoeff = abs(c0);
        for (auto it : c1) {
            chmax(maxcoeff, it.second);
        }
        for (auto it : c2) {
            chmax(maxcoeff, get<2>(it));
        }
        return maxcoeff;
    }
};
ostream & operator << (ostream & out, quadratic_pseudo_boolean_function_term_list const & a) {
    out << a.n + a.m << ' ' << a.get_terms();
    if (a.c0) {
        out << endl << 0 << ' ' << a.c0;
    }
    for (auto it : a.c1) {
        out << endl << 1 << ' ' << it.second << ' ' << it.first + 1;
    }
    for (auto it : a.c2) {
        out << endl << 2 << ' ' << get<2>(it) << ' ' << get<0>(it) + 1 << ' ' << get<1>(it) + 1;
    }
    return out;
}

int get_size_to_split_value(int value, int limit) {
    assert (value >= 0);
    int k = 1;
    while ((value - k + 1) / k > limit) ++ k;
    return k;
}

vector<int> split_value_with(int value, int k) {
    vector<int> values(k, value / k);
    REP (i, abs(value % k)) {
        values[i] += (value >= 0 ? 1 : -1);
    }
    return values;
}

vector<int> split_value(int value, int limit) {
    int split = get_size_to_split_value(abs(value), limit);
    auto values = split_value_with(value, split);
    values.erase(remove(ALL(values), 0), values.end());
    return values;
}

struct quadratic_pseudo_boolean_function_term_matrix : public quadratic_pseudo_boolean_function {
    int n;
    int m;  // or newvars
    int c0;
    vector<int> c1;  // y1 -> c
    vector<vector<int> > c2;  // y2 -> y1 -> c where y1 <= y2
    map<pair<int, int>, int> c2w;
    quadratic_pseudo_boolean_function_term_matrix(int n_) {
        n = n_;
        m = 0;
        c0 = 0;
        c1.resize(n);
        c2.resize(n);
        REP (i, n) {
            c2[i].resize(i + 1);
        }
    }

    int new_variable() {
        c1.emplace_back(0);
        c2.emplace_back(n, 0);
        return n + (m ++);
    }

    void use0(int c) {
        assert (c != 0);
        c0 += c;
    }
    void use1(int c, int y1) {
        assert (c != 0);
        c1[y1] += c;
    }
    void use2(int c, int y1, int y2) {
        assert (c != 0);
        if (y1 > y2) swap(y1, y2);
        if (y1 < n) {
            c2[y2][y1] += c;
        } else {
            auto key = make_pair(y1, y2);
            c2w[key] += c;
            if (c2w[key] == 0) {
                c2w.erase(key);
            }
        }
    }

    int get_newvars() const { assert (false); }
    int get_terms() const { assert (false); }
    int get_maxcoeff() const { assert (false); }

    void print() const {
        fprintf(stderr, "%4d:%4d\n", -1, c0);
        REP (i, n + m) {
            fprintf(stderr, "%4d:%4d", i, c1[i]);
            REP (j, min(i + 1, n)) {
                fprintf(stderr, "%4d", c2[i][j]);
            }
            fprintf(stderr, "\n");
        }
    }

    int compute_optimal_maxcoeff() const {
        int fixed_maxcoeff = 20;  // the non-zero value is a margin for safety
        REP (i, n) {
            if (c1[i] > 0) {
                chmax(fixed_maxcoeff, abs(c1[i]));
            }
            REP (j, i + 1) {
                chmax(fixed_maxcoeff, abs(c2[i][j]));
            }
        }

        int optimal_maxcoeff = -1;
        int optimal_score = -1;
        for (int delta = 0; delta < 1000; delta += 10) {
            int maxcoeff = fixed_maxcoeff + delta;
            int score = as_term_list_with_split(maxcoeff).get_score();
            if (optimal_score < score) {
                optimal_maxcoeff = maxcoeff;
                optimal_score = score;
            }
        }
        return optimal_maxcoeff;
    }

    quadratic_pseudo_boolean_function_term_list as_term_list_with_split(int maxcoeff) const {
        quadratic_pseudo_boolean_function_term_list other(n);

        // original variables of c0
        if (c0) {
            other.use0(c0);
        }

        // original positive variables of c1
        REP (i, n) if (c1[i] > 0) {
            other.use1(c1[i], i);
        }

        // original variables of c2
        REP (i, n) REP (j, i + 1) if (c2[i][j]) {
            other.use2(c2[i][j], i, j);
        }

        {  // original negative variable for c1
            // make shared new variables
            int max_c = 0;
            REP (i, n) if (c1[i] < 0) {
                chmax(max_c, abs(c1[i]));
            }
            int max_split = get_size_to_split_value(max_c, maxcoeff);
            vector<int> w(max_split);
            REP (k, max_split) {
                w[k] = other.new_variable();
            }

            // use
            REP (i, n) if (c1[i] < 0) {
                int split = get_size_to_split_value(abs(c1[i]), maxcoeff);
                auto values = split_value_with(c1[i], split);
                REP (k, split) if (values[k]) {
                    other.use2(values[k], i, w[k]);
                }
            }
        }

        // new variables
        map<int, vector<pair<int, int> > > added;
        REP3 (i, n, n + m) {
            // make split new variables
            int max_c = abs(c1[i]);
            int div = abs(c1[i]);
            REP (j, n) {
                chmax(max_c, abs(c2[i][j]));
                div = gcd(div, abs(c2[i][j]));
            }
            REP3 (j, i + 1, n + m) {
                auto it = c2w.find(make_pair(i, j));
                if (it != c2w.end()) {
                    int c = it->second;
                    chmax(max_c, abs(c));
                    div = gcd(div, abs(c));
                }
            }
            if (added.count(i)) {
                for (auto t : added[i]) {
                    int j, c; tie(j, c) = t;
                    chmax(max_c, abs(c));
                    div = gcd(div, abs(c));
                }
            }
            int split = get_size_to_split_value(max_c, maxcoeff);
            vector<int> w(split);
            REP (k, split) {
                w[k] = other.new_variable();
            }
            auto values = split_value_with(div, split);  // NOTE: here you must use `div` to avoid errors

            // use for c1
            if (c1[i]) {
                REP (k, split) if (values[k]) {
                    other.use1(values[k] * c1[i] / div, w[k]);
                }
            }

            // use for c2
            REP (j, n) if (c2[i][j]) {
                REP (k, split) if (values[k]) {
                    other.use2(values[k] * c2[i][j] / div, w[k], j);
                }
            }

            // use for c2w
            REP3 (j, i + 1, n + m) {
                auto it = c2w.find(make_pair(i, j));
                if (it != c2w.end()) {
                    int c = it->second;
                    REP (k, split) if (values[k]) {
                        added[j].emplace_back(w[k], values[k] * c / div);
                    }
                }
            }

            // use for added
            if (added.count(i)) {
                for (auto t : added[i]) {
                    int j, c; tie(j, c) = t;
                    REP (k, split) if (values[k]) {
                        other.use2(values[k] * c / div, w[k], j);
                    }
                }
            }
        }

        other.normalize();
        return other;
    }

    quadratic_pseudo_boolean_function_term_list as_term_list() const {
        return as_term_list_with_split(compute_optimal_maxcoeff());
    }

    term_t choose_nice_shuffle_for_simple_positive_reduction(term_t t) const {
        assert (t.v.size() >= 2);
        int i1 = 0;
        int i2 = 1;
        int y1 = t.v[i1];
        int y2 = t.v[i2];
        if (y1 < y2) swap(y1, y2);
        REP (j1, t.v.size()) {
            REP (j2, j1) {
                int x1 = t.v[j1];
                int x2 = t.v[j2];
                if (x1 < x2) swap(x1, x2);
                if (abs(c2[x1][x2] + t.c) < abs(c2[y1][y2] + t.c)) {
                    i1 = j1;
                    i2 = j2;
                    y1 = x1;
                    y2 = x2;
                }
            }
        }
        if (i2 == 0) swap(i1, i2);
        swap(t.v[0], t.v[i1]);
        swap(t.v[1], t.v[i2]);
        return t;
    }
};


chrono::high_resolution_clock::time_point clock_begin;
// constexpr double TLE = 30000;
constexpr double TLE = 10000;

template <class Generator>
quadratic_pseudo_boolean_function_term_list solve(int n, int k, vector<term_t> const & f, Generator & gen) {
    // in
#ifdef LOCAL
    char *path = getenv("LOG");
    if (path == nullptr) {
        cerr << "[*] N = " << n << endl;
        cerr << "[*] K = " << k << endl;
        cerr << "[*] f(1) = " << apply_all_true(f) << endl;
    }
#endif

    // prepare
    quadratic_pseudo_boolean_function_term_matrix g(n);

    // ignore
    constexpr int degree_to_ignore = 10;
    auto f1 = f;
    sort(ALL(f1), [&](term_t const & a, term_t const & b) {
        return a.v.size() < b.v.size();
    });
    int ignored_c = 0;
    while (not f1.empty() and f1.back().v.size() >= degree_to_ignore) {
        ignored_c += f1.back().c;
        f1.pop_back();
    }
    if (ignored_c) {
        vector<int> v(degree_to_ignore);
        iota(ALL(v), 0);
        f1.push_back(make_term(ignored_c, v));
    }

    // construct
    while (not f1.empty()) {
        // trivial part
        vector<term_t> f2;
        for (auto const & t : f1) {
            if (t.v.size() <= 2) {
                g.use_term(t);
            } else if (t.c < 0) {
                g.reduce_negative_monomial(t);
            } else {
                f2.push_back(t);
            }
        }
        f1.swap(f2);
        if (f1.empty()) break;

        if (bernoulli_distribution(0.5)(gen)) {
            int x_1 = f1.back().v.back();
            int w_1 = g.new_variable();
            int sum_c = 0;
            for (auto & t : f1) {
                if (find(ALL(t.v), x_1) != t.v.end()) {
                    sum_c += abs(t.c);
                    if (sum_c >= 300) break;
                    t = g.split_common_parts(t, x_1, w_1);
                }
            }
        } else {
            auto t = f1.back();
            f1.pop_back();
            shuffle(ALL(t.v), gen);
            if (bernoulli_distribution(0.99)(gen)) {
                g.simply_reduce_positive_monomial(t);
            } else {
                g.reduce_higher_order_clique(t);
            }
        }
    }

    // out
    auto g1 = g.as_term_list();
#ifdef LOCAL
    if (path == nullptr) {
        cerr << "[*] M = " << g1.get_newvars() << endl;
        cerr << "[*] L = " << g1.get_terms() << endl;
        cerr << "[*] maxcoeff = " << g1.get_maxcoeff() << endl;
        cerr << "[*] score PY = " << g1.get_score_py() << endl;
        cerr << "[*] score PZ = " << g1.get_score_pz() << endl;
        cerr << "[*] score = " << g1.get_score() << " if e_SA = 0" << endl;
        if (g1.get_newvars() < 100) {
            // cerr << "[*] f(1) = " << apply_all_true(f) << endl;
            // cerr << "[*] g(1) = " << apply_all_true_min_sa(n, m, g, gen) << endl;
            // cerr << "[*] score random = " << (int)evaluate_random_score(n, f, m, g, gen) << endl;
            // cerr << "[*] score allone = " << (int)evaluate_all_true_score(n, f, m, g, gen) << endl;
        }
    } else {
        ofstream fp(path);
        fp << "{ \"n\": " << n;
        fp << ", \"k\": " << k;
        fp << ", \"m\": " << g1.get_newvars();
        fp << ", \"l\": " << g1.get_terms();
        fp << ", \"maxcoeff\": " << g1.get_maxcoeff();
        fp << ", \"py\": " << g1.get_score_py();
        fp << ", \"pz\": " << g1.get_score_pz();
        fp << ", \"score\": " << g1.get_score();
        fp << " }" << endl;
    }
#endif
    return g1;
}


int main() {
    // init
    clock_begin = chrono::high_resolution_clock::now();
    random_device device;
    xor_shift_128 gen(device());

    // input
    int n, k; cin >> n >> k;
    vector<term_t> f(k);
    REP (i, k) {
        int d; cin >> d >> f[i].c;
        f[i].v.resize(d);
        REP (j, d) {
            cin >> f[i].v[j];
            -- f[i].v[j];
        }
    }

    // solve
    auto qpbf = solve(n, k, f, gen);

    // output
    cout << qpbf << endl;
    return 0;
}
