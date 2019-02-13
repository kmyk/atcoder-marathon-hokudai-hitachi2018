#ifndef CONSTRAINTS_FOR_HHNCCC_2018
#define CONSTRAINTS_FOR_HHNCCC_2018

const int INDEX = 1;

///////////////////////////////////////////////

const int           DEF_MIN_N[] = {   3,    3,    3}; // 変数の数最小
const int           DEF_MIN_M[] = {   1,    1,    1}; // 項の数最小
const int           DEF_MIN_D[] = {   0,    0,    0}; // 次数最小
const long long int DEF_MIN_W[] = {   1,    1,    1}; // 係数幅の絶対値最小

///////////////////////////////////////////////

const int           DEF_MAX_N[] = {  50,  300,   10}; // 変数の数最大
const int           DEF_MAX_M[] = {  50, 1000, 1024}; // 項の数最大
const int           DEF_MAX_D[] = {  50,    6,    6}; // 次数最大
const long long int DEF_MAX_W[] = { 100,  100,  100}; // 係数幅の絶対値最大

///////////////////////////////////////////////

const int MAX_S_OUT = 3000;

const int MIN_L_OUT = 1;

const int MIN_D_OUT = 0;
const int MAX_D_OUT = 2;

const long long int MIN_C_OUT = 1;
const long long int MAX_C_OUT = (1LL << 31) - 1;

#endif
