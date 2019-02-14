#include "../common/constraints.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <climits>
#include <random>
#include <cstdio>
#include <utility>
#include <set>

//////// for evaluation ////////
std::mt19937_64 mt;

const int repnum = 5;
const long long int INF = 1LL << 60;

//////// error code ////////
enum ErrorCode {
  OUT_OF_RANGE_OUT            = 0,
  NOT_UNIQUE_OUT              = 1,
  NOT_STRICTLY_INCREASING_OUT = 2,
  DO_NOT_APPEAR_VARIABLE      = 3,
  INVALID_FUNCTION_VALUE      = 4,
};

unsigned long long int str2ull(char *str) {
  char* end_point;
  unsigned long long int ret_number = strtoull(str, &end_point, 10);
  if(*end_point != '\0') {
    fprintf(stderr, "Error: invalid input: \"%s\"\n", str);
    exit(1);
  }
  return ret_number;
}

/* sa.h */
// a function performing simulated annealing
void sa(std::vector<std::vector<std::pair<long long int, long long int> > > &J,
	std::vector<long long int> &h,
	std::vector<int> &spins,
	int fixedspin,
	int nsweeps = 10000,
	double inv_temp = 0.00001,
	double cool_coeff = 1.0023052380778996,
	bool verbose = false);

/* sa.cpp */
// energy function for Ising model
long long int energy(std::vector<std::vector<std::pair<long long int, long long int> > >  &J,
		     std::vector<long long int> &h,
		     std::vector<int> &spins){
  int N = J.size();
  long long int E = 0;
  for(int i=0; i<N; ++i){
    E += h[i] *spins[i];
    for(auto next : J[i]){
      long long int &j = next.first;
      long long int &Jij = next.second;
      E += spins[i] *Jij *spins[j]; 
    }
  }
  return E;
}

// simulated annealing function
void sa(std::vector<std::vector<std::pair<long long int, long long int> > > &J,
	std::vector<long long int> &h,
	std::vector<int> &spins,
	int fixedspin,
	int nsweeps,
	double inv_temp,
	double cool_coeff,
	bool verbose){
  /* This function performs simulated annealing using an exponential schedule
       
     Spin System:
     ------------
     N ... number of spins (length of vector)
     J ... interactions; coefficients of quadratic terms stored as adjacent list
     h ... external magnetic fields; array of length N
     spins ... spins; array of length N (values: 0 or 1)
       
     Schedule:
     ---------
     nsweeps ... number of temperature steps
     inv_temp ... inverse temperature (initial value)
     cool_coeff ... a multiplier which slowly increases the inverse temperature
  */

  int N = J.size();
    
  // initialize random number generator
  std::uniform_int_distribution<int> zero_one(0, 1);
  std::uniform_real_distribution<double> uniform(0, 1);
    
  // initialize spins
  for(int i=fixedspin; i<N; ++i){
    spins[i] = zero_one(mt);
  }
    
  // iterature through the temperature
  for(int i=0; i<nsweeps; ++i){
    if((verbose)&&(i%100==0)){
      printf("sweep: %d / %d \n", i+1, nsweeps);
      printf("energy: %lld \n", energy(J, h, spins));
    }
    //iterate through the spin variables and try flipping each spin once
    for(int j=fixedspin; j<N; ++j){
      // compute the change in energy
      long long int deltaE = h[j];
      for(auto next : J[j]){
	long long int &k = next.first;
	long long int &Jjk = next.second;
	deltaE += 2 * Jjk * spins[k];//
      }
      if(spins[j] == 1){
	deltaE = -deltaE;
      }
	    
      // decide whether or not to flip the spin
      if(deltaE < 0){
	// flip
	spins[j] = 1 -spins[j];
      }else if(uniform(mt) < exp(-inv_temp*deltaE)){
	// flip
	spins[j] = 1 -spins[j];	
      }
    }
    // update the inverse temperature
    inv_temp *= cool_coeff;
  }
}

// substitute spins into pseudo-boolean function
long long int substitution(std::vector<int> &spins,
			   std::vector<std::vector<long long int> > &pbf){
  long long int res = 0;
  for(const std::vector<long long int> &term : pbf){
    bool nonzero = true;
    for(unsigned int i=1; i<term.size(); ++i){
      if(spins[term[i]] == 0){
	nonzero = false;
	break;
      }
    }
    if(nonzero){
      res += term[0];
    }
  }
  return res;
}

//call simulated-annealing module for minimizing quadratized pseudo-boolean function
long long int minimize_qpbf(int N, 
			    std::vector<int> spins,
			    std::vector<std::vector<long long int> > &qpbf){
  std::vector<std::vector<std::pair<long long int, long long int> > > J(N);
  std::vector<long long int> h(N, 0);
  for(const std::vector<long long int> &term : qpbf){
    const long long int &coeff = term[0];
    int deg = term.size() - 1;
    if(deg == 1){
      h[term[1]] = 2 * coeff;
    }
    if(deg == 2){
      J[term[1]].emplace_back(term[2], coeff);
      J[term[2]].emplace_back(term[1], coeff);
    }
  }
  int n = spins.size();
  spins.resize(N);
  long long int res = INF;
  for(int i=0; i<repnum; ++i){
    sa(J, h, spins, n);
    res = std::min(res, substitution(spins, qpbf));
  }
  return res;
}

int main(int argc, char **argv){
  const int COMMAND_ARG    = 0;
  const int INPUT_ARG      = 1;
  const int OUTPUT_ARG     = 2;
  const int SEED_ARG       = 3;
  const int NUM_OF_ARGS    = 4;

  if(argc != NUM_OF_ARGS){
    std::cerr << "Usage : " << argv[COMMAND_ARG] << " <input-filename> <output-filename> <seed>" << std::endl;
    return 1;
  }

  // random seed (for sa)
  unsigned long long int SEED = str2ull(argv[SEED_ARG]);
  mt.seed(SEED);

  try{
    //read HOBO
    std::ifstream input(argv[INPUT_ARG], std::ios::in);
    std::vector< std::vector<long long int> > hobo;
    int hn, hk;
    {
      // number of variables, number of terms
      input >> hn >> hk;
      hobo.resize(hk);

      // terms information
      for(int i=0; i<hk; ++i){
	// degree
	int d; input >> d;

	// coefficient
	long long int c; input >> c;
	hobo[i].push_back(c);

	std::vector<long long int> vars;
	// variable id
	for(int j=0; j<d; ++j){
	  long long int v; input >> v;
	  
	  v -= INDEX;
	  hobo[i].push_back(v);
	  vars.push_back(v);
	}
      }
    }
    input.close();
	
    // read QUBO (with checking output)
    std::ifstream output(argv[OUTPUT_ARG], std::ios::in);
    std::vector<std::vector<long long int> > qubo;
    int qs, ql;
    {
      // check whether the combination of variables is unique
      std::set< std::vector<long long int> > unique_vec;

      // number of variables, number of terms
      output >> qs >> ql;
      const int MIN_S_OUT = hn;
      const int MAX_L_OUT = 1 + qs + qs * (qs - 1) / 2;
      if(qs < MIN_S_OUT or MAX_S_OUT < qs) throw OUT_OF_RANGE_OUT;
      if(ql < MIN_L_OUT or MAX_L_OUT < ql) throw OUT_OF_RANGE_OUT;
      qubo.resize(ql);

      // appearing check for all additional variables
      std::vector<int> appeared(qs);

      // terms information
      for(int i=0; i<ql; ++i){
	// degree
	int d; output >> d;
	if(d < MIN_D_OUT or MAX_D_OUT < d) throw OUT_OF_RANGE_OUT;
	
	// coefficient
	long long int c; output >> c;
	if(std::abs(c) < MIN_C_OUT or MAX_C_OUT < std::abs(c)) throw OUT_OF_RANGE_OUT;
	qubo[i].push_back(c);

	std::vector<long long int> vars;
	// variable id
	long long int last_id = -1;
	for(int j=0; j<d; ++j){
	  long long int v; output >> v;
	  if(v < INDEX or qs + INDEX <= v) throw OUT_OF_RANGE_OUT;

	  v -= INDEX;
	  if(last_id >= v) throw NOT_STRICTLY_INCREASING_OUT;
	  last_id = v;
	  
	  qubo[i].push_back(v);
	  vars.push_back(v);
	  appeared[v] = true;
	}

	if(unique_vec.count(vars)) throw NOT_UNIQUE_OUT;
	unique_vec.insert(vars);
      }

      for(int i=hn; i<qs; i++) {
	if(!appeared[i]) throw DO_NOT_APPEAR_VARIABLE;
      }
    }
    output.close();
    
    using Formula = std::vector< std::vector<long long int> >;
    auto calc_score = [&](const std::string & title, std::vector<int> X, int HN, int QS, int QL, Formula HOBO, Formula QUBO) {
      long long int fxval = substitution(X, HOBO);
      long long int gxmin = minimize_qpbf(QS, X, QUBO);
      double e_SA = (double)(gxmin - fxval);
      double T = 100.0;
      
      double M_newvars = QS - HN;
      double L_terms   = QL;
      
      long long int maxC = 0;
      for(const std::vector<long long int> &term : QUBO){
	// do not include constant term
	if(term.size() == 1) continue;
	maxC = std::max(maxC, std::abs(term[0]));
      }
      
      // constants for evaluation
      const double A = 10000;
      const double B = 5;
      const double E = 10000;
      
      double point = 0;
      if(e_SA < 0){
        std::cerr << title << std::endl;
        std::cerr << "e_SA     : " << e_SA << std::endl;
	return -1.0;
      }
      else{
	double CONST_Y = 1000;
	double CONST_Z = 1000;
	
	double PX = E * (1.0 - (std::min(e_SA, T) / T));
	double PY = (CONST_Y) / (B * M_newvars + L_terms + CONST_Y);
	double PZ = (CONST_Z) / (maxC + CONST_Z);
	point = A * PX * PY * PZ;
      }
      

      std::cerr << title << std::endl;
      std::cerr << "e_SA     : " << e_SA << std::endl;
      std::cerr << "newvars  : " << M_newvars << std::endl;
      std::cerr << "terms    : " << L_terms << std::endl;
      std::cerr << "maxcoeff : " << maxC << std::endl;
      std::cerr << "--------------------------" << std::endl;
      std::cerr << "score    : " << (int)point << std::endl << std::endl;
      
      return point;
    };

    // calculate score
    std::vector<int> y(hn), z(hn, 1);
    std::uniform_int_distribution<int> zero_one(0, 1);
    for(int i=0; i<hn; ++i){
      y[i] = zero_one(mt);
    }

    double point1 = calc_score("random", y, hn, qs, ql, hobo, qubo);
    double point2 = calc_score("all 1", z, hn, qs, ql, hobo, qubo);

    if(point1 < 0 or point2 < 0) throw INVALID_FUNCTION_VALUE;
    double point = (point1 + point2) / 2.0;
    printf("total point = %d\n", (int)point);
    return 0;
  }
  catch(ErrorCode mode){
    switch(mode) {
    case OUT_OF_RANGE_OUT:
      fprintf(stderr, "Error: out of range (output)\n");
      break;
    case NOT_UNIQUE_OUT:
      fprintf(stderr, "Error: the combination of variables is not unique (output)\n");
      break;
    case NOT_STRICTLY_INCREASING_OUT:
      fprintf(stderr, "Error: not strictly increasing order (output)\n");
      break;
    case DO_NOT_APPEAR_VARIABLE:
      fprintf(stderr, "Error: additional variable is not appeared (output)\n");
      break;
    case INVALID_FUNCTION_VALUE:
      fprintf(stderr, "Error: function value is invalid (function value of \'g\' is smaller than that of \'f\' (output)\n");
      break;
    }
    return 1;
  }
  return 1;
}
