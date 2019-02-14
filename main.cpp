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

pair<int, vector<term_t> > remove_unused_newvars(int n, int m, vector<term_t> g) {
    // mark
    vector<bool> used(m);
    for (auto const & t : g) {
        for (int v_i : t.v) {
            if (v_i >= n) {
                used[v_i - n] = true;
            }
        }
    }

    // compress
    vector<int> rename(m, -1);
    int updated_m = 0;
    REP (i, m) {
        if (used[i]) {
            rename[i] = n + updated_m ++;
        }
    }

    // apply
    for (auto & t : g) {
        for (int & v_i : t.v) {
            if (v_i >= n) {
                v_i = rename[v_i - n];
            }
        }
    }
    return make_pair(updated_m, g);
}

vector<term_t> merge_terms(int l, vector<term_t> const & g) {
    // collect into buckets
    int c0 = 0;
    vector<int> c1(l);
    auto c2 = vectors(l, l, int());
    for (auto const & t : g) {
        if (t.v.size() == 0) {
            c0 += t.c;
        } else if (t.v.size() == 1) {
            c1[t.v[0]] += t.c;
        } else if (t.v.size() == 2) {
            c2[t.v[0]][t.v[1]] += t.c;
            c2[t.v[1]][t.v[0]] += t.c;
        } else {
            assert (false);
        }
    }

    // reconstruct
    vector<term_t> h;
    if (c0) {
        h.push_back(make_term(c0, {}));
    }
    REP (i, l) {
        if (c1[i]) {
            h.push_back(make_term(c1[i], { i }));
        }
    }
    REP (j, l) {
        REP (i, j) {
            if (c2[i][j]) {
                h.push_back(make_term(c2[i][j], { i, j }));
            }
        }
    }
    return h;
}

int get_size_to_split_value(int value, int limit) {
    assert (value >= 0);
    int k = 1;
    while ((value - k + 1) / k > limit) ++ k;
    return k;
}

vector<int> split_value_with(int value, int k) {
    vector<int> values(k, value / k);
    values.back() += value % k;
    return values;
}

double evaluate_score_px(int m, vector<term_t> const & g, int delta) {
    constexpr int e = 10000;
    constexpr int t = 100;
    if (delta < 0) return - INFINITY;
    return e * (1 - min<double>(t, delta) / t);
}
double evaluate_score_py(int m, vector<term_t> const & g) {
    constexpr int b = 5;
    int l = g.size();
    return 1000 / (b * m + l + 1000.0);
}
double evaluate_score_pz(int m, vector<term_t> const & g) {
    int maxcoeff = get_maxcoeff(g);
    return 1000 / (maxcoeff + 1000.0);
}
double evaluate_score(int m, vector<term_t> const & g, int delta) {
    constexpr int a = 10000;
    double px = evaluate_score_px(m, g, delta);
    double py = evaluate_score_py(m, g);
    double pz = evaluate_score_pz(m, g);
    return a * px * py * pz;
}

template <class Generator>
double evaluate_all_true_score(int n, vector<term_t> const & f, int m, vector<term_t> const & g, Generator & gen) {
    int delta = apply_all_true_min_sa(n, m, g, gen) - apply_all_true(f);
    double pa = evaluate_score(m, g, delta);
    return pa;
}

template <class Generator>
double evaluate_random_score(int n, vector<term_t> const & f, int m, vector<term_t> const & g, Generator & gen) {
    double pb = 0;
    constexpr int width = 10;
    REP (iteration, width) {
        auto x = generate_random_vector(n, gen);
        int delta = apply_vector_min_sa(m, g, x, gen) - apply_vector(f, x);
        pb += evaluate_score(m, g, delta) / width;
    }
    return pb;
}

chrono::high_resolution_clock::time_point clock_begin;
// constexpr double TLE = 30000;
constexpr double TLE = 10000;


template <class Generator>
pair<int, vector<term_t> > solve(int n, int k, vector<term_t> f, Generator & gen) {
    // in
#ifdef LOCAL
    char *path = getenv("LOG");
    if (path == nullptr) {
        cerr << "[*] N = " << n << endl;
        cerr << "[*] K = " << k << endl;
        cerr << "[*] f(1) = " << apply_all_true(f) << endl;
    }
#endif
    sort(ALL(f), [&](term_t const & a, term_t const & b) {
        return a.v.size() < b.v.size();
    });
    for (auto & t : f) {
        sort(ALL(t.v));
    }

    // prepare
    int m = 0;
    vector<term_t> g;

    map<int, int> coeff1;
    map<pair<int, int>, int> coeff2;

    auto use0 = [&](int c) {
        g.push_back(make_term(c, {}));
    };
    auto use1 = [&](int c, int y1) {
        coeff1[y1] += c;
        g.push_back(make_term(c, { y1 }));
    };
    auto use2 = [&](int c, int y1, int y2) {
        coeff2[make_pair(y1, y2)] += c;
        coeff2[make_pair(y2, y1)] += c;
        g.push_back(make_term(c, { y1, y2 }));
    };

    constexpr int maxcoeff = 200;

    // construct
    for (auto const & t : f) {

        if (t.v.size() <= 2) {
            // the trivial case
            g.push_back(t);

        } else if (t.c < 0) {
            // the simple quadratization of negative monomials
            int d = t.v.size();
            int split = get_size_to_split_value(abs(t.c * (d - 1)), maxcoeff);
            for (int c : split_value_with(t.c, split)) {
                int w1 = n + (m ++);
                use1(- c * (d - 1), w1);
                for (int x1 : t.v) {
                    use2(c, w1, x1);
                }
            }

        } else {
            // the simple quadratization of positive monomials
            int d = t.v.size();
            auto v = t.v;

            // choose a nice pair (x1, x2)
            constexpr int total = 100;
            REP (iteration, total) {
                shuffle(ALL(v), gen);
                auto key = make_pair(v[0], v[1]);
                if (not coeff2.count(key) or abs(coeff2[key] + t.c) < maxcoeff * (1 + (double) iteration / total)) {
                    break;
                }
            }

            // c
            use0(t.c);
            // - c (1 - x1)
            use0(- t.c);
            use1(t.c, v[0]);
            // - c x1 (1 - x2)
            use1(- t.c, v[0]);
            use2(t.c, v[0], v[1]);
            REP3 (i, 2, d) {
                // -c x1 x2 .. x{i - 1} (1 - xi)
                int split = get_size_to_split_value(abs(t.c * i), maxcoeff);
                for (int c : split_value_with(t.c, split)) {
                    int wi = n + (m ++);
                    use1(c * i, wi);
                    use1(- c, wi);
                    use2(c, wi, v[i]);
                    REP (j, i) {
                        use2(- c, wi, v[j]);
                    }
                }
            }
        }
    }
    tie(m, g) = remove_unused_newvars(n, m, g);
    g = merge_terms(n + m, g);
    assert (not g.empty());

    // out
#ifdef LOCAL
    if (path == nullptr) {
        cerr << "[*] M = " << m << endl;
        cerr << "[*] L = " << g.size() << endl;
        cerr << "[*] maxcoeff = " << get_maxcoeff(g) << endl;
        cerr << "[*] score PY = " << evaluate_score_py(m, g) << endl;
        cerr << "[*] score PZ = " << evaluate_score_pz(m, g) << endl;
        cerr << "[*] score = " << evaluate_score(m, g, 0) << " if e_SA = 0" << endl;
        if (m < 100) {
            cerr << "[*] f(1) = " << apply_all_true(f) << endl;
            cerr << "[*] g(1) = " << apply_all_true_min_sa(n, m, g, gen) << endl;
            cerr << "[*] score random = " << (int)evaluate_random_score(n, f, m, g, gen) << endl;
            cerr << "[*] score allone = " << (int)evaluate_all_true_score(n, f, m, g, gen) << endl;
        }
    } else {
        ofstream fp(path);
        fp << "{ \"n\": " << n;
        fp << ", \"k\": " << k;
        fp << ", \"m\": " << m;
        fp << ", \"l\": " << g.size();
        fp << ", \"maxcoeff\": " << maxcoeff;
        fp << ", \"py\": " << evaluate_score_py(m, g);
        fp << ", \"pz\": " << evaluate_score_pz(m, g);
        fp << ", \"score\": " << evaluate_score(m, g, 0);
        fp << " }" << endl;
    }
#endif
    return make_pair(m, g);
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
    int m; vector<term_t> g;
    tie(m, g) = solve(n, k, f, gen);

    // output
    cout << n + m << ' ' << g.size() << endl;
    for (auto const & t : g) {
        cout << t.v.size() << ' ' << t.c;
        for (int v_j : t.v) {
            cout << ' ' << v_j + 1;
        }
        cout << endl;
    }
    return 0;
}
