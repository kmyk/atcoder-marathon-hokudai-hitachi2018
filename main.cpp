#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
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

int get_max_c(vector<term_t> const & f) {
    int max_c = 0;
    for (auto const & t : f) {
        if (t.v.empty()) continue;  // ignore constants
        chmax(max_c, t.c);
    }
    return max_c;
}

int get_constant_term(vector<term_t> const & f) {
    int sum_c = 0;
    for (auto const & t : f) {
        if (t.v.empty()) {
            sum_c += t.c;
        }
    }
    return sum_c;
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
    vector<bool> w = generate_random_vector(m, gen);
    int value = apply_argumented_vector(g, x, w);
    int min_value = value;

    int total = 10 * m;
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

int apply_vector_min(vector<term_t> const & g, vector<bool> const & x, int m) {
    int n = g.size() - m;
    int min_value = INT_MAX;
    REP (w, 1 << m) {
        int value = 0;
        for (auto const & t : g) {
            for (int i : t.v) {
                if (not (i < n ? x[i] : w & (1 << (i - n)))) {
                    goto next;
                }
            }
            value += t.c;
    next: ;
        }
        chmin(min_value, value);
    }
    return min_value;
}

vector<term_t> extract_low_degree_terms(vector<term_t> const & f) {
    vector<term_t> g;
    for (auto const & t : f) {
        if (t.v.size() <= 2) {
            g.push_back(t);
        }
    }
    return g;
}

vector<term_t> extract_initial_polynomial(vector<term_t> const & f) {
    vector<term_t> g;
    for (auto const & t : f) {
        if (t.v.size() <= 2) {
            g.push_back(t);
        } else {
            int d = t.v.size();
            REP (j, d) REP (i, j) {
                g.push_back(make_term(t.c / (d * (d - 1) / 2), { i, j }));
            }
        }
    }
    return g;
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

double evaluate_relaxed_score1(int m, int l, int max_c, int delta) {
    // constexpr double a = 10000;
    constexpr double b = 5;
    // constexpr double e = 10000;
    constexpr double t = 100;
    double px = delta >= 0 ? (1 - min<double>(t, delta) / t) : 0;
    double px1 = (1 - abs(delta) / t);
    double py = 1000 / (b * m + l + 1000.0);
    double pz = 1000 / (max_c + 1000.0);
    int penalty = (delta < 0 ? - 10000 : 0);
    return (px + 0.1 * px1) * py * pz + penalty;
}

template <class Generator>
double evaluate_relaxed_score(vector<term_t> const & f, int m, vector<term_t> const & g, Generator & gen) {
    int n = g.size() - m;
    int l = g.size();
    int max_c = get_max_c(g);
    double pa = evaluate_relaxed_score1(m, l, max_c, apply_all_true(g) - apply_all_true(f));
    double pb = 0;
    constexpr int width = 300;
    xor_shift_128 fixed;
    REP (iteration, width) {
        constexpr int margin = 0;
        auto x = generate_random_vector(n, iteration < width ? fixed : gen);
        int delta = apply_vector_min(g, x, m) - apply_vector(f, x) - margin;
        pb += evaluate_relaxed_score1(m, l, max_c, delta) / width;
    }
    return (pa + pb) / 2;
}

chrono::high_resolution_clock::time_point clock_begin;
// constexpr double TLE = 30000;
constexpr double TLE = 10000;


template <class Generator>
pair<int, vector<term_t> > solve(int n, int k, vector<term_t> f, Generator & gen) {
    // in
    cerr << "[*] N = " << n << endl;
    cerr << "[*] K = " << k << endl;
    cerr << "[*] f(1) = " << apply_all_true(f) << endl;
    sort(ALL(f), [&](term_t const & a, term_t const & b) {
        return a.v.size() < b.v.size();
    });
    for (auto & t : f) {
        sort(ALL(t.v));
    }

    // body
    int m = 0;
    vector<term_t> g;

    auto use0 = [&](int c) {
        g.push_back(make_term(c, {}));
    };
    auto use1 = [&](int c, int y1) {
        g.push_back(make_term(c, { y1 }));
    };
    auto use2 = [&](int c, int y1, int y2) {
        g.push_back(make_term(c, { y1, y2 }));
    };

    auto define = [&](int x1, int x2, int w1) {
        constexpr int max_c = 100;
        constexpr int c = 2 * max_c + 10;
        constexpr int b = c + max_c + 5;
        constexpr int a = 2 * c - b;
        assert (b >= 0);  // when x1 = x2 = 0
        assert (b >= c);  // when x1 = 1 and x2 = 0, or when x1 = 0 and x2 = 1
        assert (b <= 2 * c and a + b - 2 * c == 0);  // when x1 = x2 = 1
        assert (b - c - max_c >= 0);  // when - k w1 is used at an other place
        assert (b - 2 * c + max_c <= 0);  // when + k w1 is used at an other place
        use2(a, x1, x2);
        use1(b, w1);
        use2(- c, x1, w1);
        use2(- c, x2, w1);
    };
    auto squash = [&](int x1, int x2) {
        int w1 = n + (m ++);
        define(x1, x2, w1);
        // cerr << "[*] use " << w1 + 1 << " as " << x1 + 1 << " " << x2 + 1 << endl;
        return w1;
    };

    for (auto const & t : f) {
        if (t.v.size() <= 2) {
            g.push_back(t);
        } else if (t.c < 0) {
            int d = t.v.size();
            int w1 = n + (m ++);
            use1(- t.c * (d - 1), w1);
            for (int x1 : t.v) {
                use2(t.c, w1, x1);
            }
            // cerr << "[*] use " << w1 + 1 << " for the t with c = " << t.c << endl;
        } else {
            int d = t.v.size();
            auto v = t.v;
            shuffle(ALL(v), gen);
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
                int wi = n + (m ++);
                use1(t.c * i, wi);
                use1(- t.c, wi);
                use2(t.c, wi, v[i]);
                REP (j, i) {
                    use2(- t.c, wi, v[j]);
                }
            }
        }
    }
    tie(m, g) = remove_unused_newvars(n, m, g);
    g = merge_terms(n + m, g);

    // out
    cerr << "[*] M = " << m << endl;
    cerr << "[*] L = " << g.size() << endl;
    cerr << "[*] f(1) = " << apply_all_true(f) << endl;
    if (m <= 100) {
        cerr << "[*] g(1) = " << apply_all_true_min_sa(n, m, g, gen) << endl;
    }
    // cerr << "[*] score = " << highscore << endl;
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
