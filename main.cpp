#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
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

struct term_t {
    int c;
    vector<int> v;
};
term_t make_term(int c, vector<int> const & v) {
    return (term_t) { c, v };
}

pair<int, vector<term_t> > solve(int n, int k, vector<term_t> const & f) {
    int m = 0;
    vector<term_t> g;
    g.push_back(make_term(10000, {}));
    return make_pair(m, g);
}

int main() {
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
    tie(m, g) = solve(n, k, f);

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
