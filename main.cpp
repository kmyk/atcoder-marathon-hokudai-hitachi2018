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

vector<term_t> normalize_polynomial(int n, int m, vector<term_t> const & g) {
    int c0 = 0;
    vector<int> c1(n + m);
    auto c2 = vectors(n + m, n + m, int());
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
    vector<term_t> h;
    if (c0) {
        h.push_back(make_term(c0, {}));
    }
    REP (i, n + m) {
        if (c1[i]) {
            h.push_back(make_term(c1[i], { i }));
        }
    }
    REP (i, n + m) {
        REP (j, i) {
            if (c2[i][j]) {
                h.push_back(make_term(c2[i][j], { j, i }));
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
    double px = (1 - (double)abs(delta) / t);
    double py = 1000 / (b * m + l + 1000.0);
    double pz = 1000 / (max_c + 1000.0);
    int penalty = (delta < 0 ? - 10000 : 0);
    return px * py * pz + penalty;
}

template <class Generator>
double evaluate_relaxed_score(vector<term_t> const & f, int m, vector<term_t> const & g, Generator & gen_) {
    int n = g.size() - m;
    int l = g.size();
    int max_c = get_max_c(g);
    double pa = evaluate_relaxed_score1(m, l, max_c, apply_all_true(g) - apply_all_true(f));
    double pb = 0;
    constexpr int width = 100;
    xor_shift_128 gen;
    REP (iteration, width) {
        auto x = generate_random_vector(n, gen);
        int delta = apply_vector_min(g, x, m) - apply_vector(f, x) - 10;
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

    // body
    constexpr int m = 0;
    vector<term_t> g;
    g.push_back(make_term(get_constant_term(f), {}));
    if (g.back().c == 0) {
        g.pop_back();
    }
    double score = evaluate_relaxed_score(f, m, g, gen);
    cerr << "[*] score = " << score << endl;

    vector<term_t> best_g = g;
    double highscore = score;

    double temperature = 1;
    for (unsigned iteration = 0; ; ++ iteration) {
        chrono::high_resolution_clock::time_point clock_end = chrono::high_resolution_clock::now();
        auto cnt = chrono::duration_cast<chrono::milliseconds>(clock_end - clock_begin).count();
        if (cnt >= TLE * 0.99) {
            cerr << "iteration = " << iteration << ": done" << endl;
            break;
        }
        temperature = 1 - cnt / TLE;

        // make a neighbor
        auto h = g;
        if (not h.empty() and bernoulli_distribution(0.1)(gen)) {
            int i = uniform_int_distribution<int>(0, h.size() - 1)(gen);
            if (bernoulli_distribution(0.3)(gen)) {
                swap(h[i], h.back());
                h.pop_back();
            } else {
                int c = uniform_int_distribution<int>(-20, 20)(gen);
                if (c == 0) continue;
                h[i].c += c;
                if (h[i].c == 0) continue;
            }
        } else {
            int c = uniform_int_distribution<int>(-50, 100)(gen);
            if (c == 0) continue;
            int d = uniform_int_distribution<int>(1, 2)(gen);
            vector<int> v(d);
            REP (i, d) {
                while (true) {
                    int v_i = uniform_int_distribution<int>(0, n + m - 1)(gen);
                    if (not count(ALL(v), v_i)) {
                        v[i] = v_i;
                        break;
                    }
                }
            }
            term_t t = make_term(c, v);
            h.push_back(t);
            h = normalize_polynomial(n, m, h);
        }

        double delta = evaluate_relaxed_score(f, m, h, gen) - score;
        constexpr double boltzmann = 20;
        if (delta >= 0 or bernoulli_distribution(exp(boltzmann * delta / temperature))(gen)) {
            g = h;
            score += delta;
// if (delta < 0) cerr << "[*] iteration = " << iteration << ": force = " << score << endl;
            if (highscore < score) {
                highscore = score;
                best_g = g;
                cerr << "[+] iteration = " << iteration << ": highscore = " << highscore << endl;
            }
        }
    }

    // out
    cerr << "[*] M = " << m << endl;
    cerr << "[*] L = " << best_g.size() << endl;
    cerr << "[*] g(1) = " << apply_all_true(best_g) << endl;
    cerr << "[*] score = " << highscore << endl;
    return make_pair(m, best_g);
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
