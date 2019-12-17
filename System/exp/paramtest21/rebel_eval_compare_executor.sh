
#mkdir -p logs/eval_compare


#./exp/paramtest21/rebel_eval_compare.sh "baseline" f f 1 1 1.0 1.0 300 1.0 24 > logs/eval_compare/baseline.log
# WER: 17.47
#./exp/paramtest21/rebel_eval_compare.sh "baseline" f f 1 1 1.0 1.0 300 1.0 96 > logs/eval_compare/baseline-96.log
# WER: 17.29
#./exp/paramtest21/rebel_eval_compare.sh "baseline" f f 1 1 1.0 1.0 300 1.0 128 > logs/eval_compare/baseline-128.log
# 17.22
#./exp/paramtest21/rebel_eval_compare.sh "baseline" f f 1 1 1.0 1.0 300 1.0 256 > logs/eval_compare/baseline-256.log
# 17.47
#./exp/paramtest21/rebel_eval_compare.sh "baseline" f f 1 1 1.0 1.0 300 1.0 561 > logs/eval_compare/baseline-all.log
# 17.21

#./exp/paramtest21/rebel_eval_compare.sh "beamSearch" f f 2   1 1.0 1.0 300 1.0 24 > logs/eval_compare/beamSearch2.log
# 49.35
#./exp/paramtest21/rebel_eval_compare.sh "beamSearch" f f 16  1 1.0 1.0 300 1.0 24 > logs/eval_compare/beamSearch16.log
# 78.9
#./exp/paramtest21/rebel_eval_compare.sh "beamSearch" f f 64  1 1.0 1.0 300 1.0 24 > logs/eval_compare/beamSearch64.log
# 78.5
#./exp/paramtest21/rebel_eval_compare.sh "beamSearch" f f 128 1 1.0 1.0 300 1.0 24 > logs/eval_compare/beamSearch128.log
# 79
#./exp/paramtest21/rebel_eval_compare.sh "beamSearch" f f 512 1 1.0 1.0 300 1.0 24 > logs/eval_compare/beamSearch512.log
# 77

#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 2   1 1.0 1.0 300 1.0 24 > logs/eval_compare/WordCTC2.log
# 18.1
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 16  1 1.0 1.0 300 1.0 24 > logs/eval_compare/WordCTC16.log
# 14.58
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 64  1 1.0 1.0 300 1.0 24 > logs/eval_compare/WordCTC64.log
# 14.3
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0 24 > logs/eval_compare/WordCTC128.log
# 14.2
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 512 1 1.0 1.0 300 1.0 24 > logs/eval_compare/WordCTC512.log
# 14.13
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 1024 1 1.0 1.0 300 1.0 24 > logs/eval_compare/WordCTC1024.log
# 14.16
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 2048 1 1.0 1.0 300 1.0 24 > logs/eval_compare/WordCTC2048.log
# 14.1

## Batch size change for the Evaluation...
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0  1 > logs/eval_compare/WordCTC128-b1.log
# 33.26
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0  2 > logs/eval_compare/WordCTC128-b2.log
# 19.46
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0  4 > logs/eval_compare/WordCTC128-b4.log
# 15.38
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0  8 > logs/eval_compare/WordCTC128-b8.log
# 14.84
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0 24 > logs/eval_compare/WordCTC128.log
# 14.2
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0 48 > logs/eval_compare/WordCTC128-b48.log
# 14.13
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0 96 > logs/eval_compare/WordCTC128-b96.log
# 13.96
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0 128 > logs/eval_compare/WordCTC128-b128.log
# 14.08
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0 256 > logs/eval_compare/WordCTC128-b256.log
# 14.01
#./exp/paramtest21/rebel_eval_compare.sh "WordCTC" "t" "f" 128 1 1.0 1.0 300 1.0 561 > logs/eval_compare/WordCTC128-b561.log
# OOM - Out of main memory 16 GB main and 16GB swap

#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 2   0.05 0.15 300 1.0 24 > logs/eval_compare/lm2.log
# 71.11
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 16  0.05 0.15 300 1.0 24 > logs/eval_compare/lm16.log
# 29.61
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 64  0.05 0.15 300 1.0 24 > logs/eval_compare/lm64.log
# 24.8
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.15 300 1.0 24 > logs/eval_compare/lm128.log
# 23.72
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 512 0.05 0.15 300 1.0 24 > logs/eval_compare/lm512.log
# 22.47
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 1024 0.05 0.15 300 1.0 24 > logs/eval_compare/lm1024.log
# 22.1
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 2048 0.05 0.15 300 1.0 24 > logs/eval_compare/lm2048.log
# 21.88
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 4096 0.05 0.15 300 1.0 24 > logs/eval_compare/lm4096.log
# 21.6

## Batch size change for the Evaluation... baseline is 
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.15 300 1.0 24 > logs/eval_compare/lm128.log
# 23.72
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.15 300 1.0 1 > logs/eval_compare/lm128-b1.log
# 32.15
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.15 300 1.0 96 > logs/eval_compare/lm128-b1.log
# 22.99

## Language model alpha
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.03 300 1.0 24 > logs/eval_compare/lm128-a03.log
# 44.17
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.05 300 1.0 24 > logs/eval_compare/lm128-a05.log
# 37.24
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.10 300 1.0 24 > logs/eval_compare/lm128-a10.log
# 27.7
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.15 300 1.0 24 > logs/eval_compare/lm128.log
# 23.72
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.175 300 1.0 24 > logs/eval_compare/lm128-a17-5.log
# 22.87
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.19 300 1.0 24 > logs/eval_compare/lm128-a19.log
# 22.26
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.20 300 1.0 24 > logs/eval_compare/lm128-a20.log
# 22.43
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.21 300 1.0 24 > logs/eval_compare/lm128-a21.log
# 22.27
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.25 300 1.0 24 > logs/eval_compare/lm128-a25.log
# 23.30
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.30 300 1.0 24 > logs/eval_compare/lm128-a30.log
# 26.36
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.40 300 1.0 24 > logs/eval_compare/lm128-a40.log
# 40.64

## Language model beta
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.00 0.15 300 1.0 24 > logs/eval_compare/lm128-be01.log
# 23.63
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.01 0.15 300 1.0 24 > logs/eval_compare/lm128-be01.log
# 21.6
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.03 0.15 300 1.0 24 > logs/eval_compare/lm128-be03.log
# 23.5
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.05 0.15 300 1.0 24 > logs/eval_compare/lm128.log
# 23.72
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.10 0.15 300 1.0 24 > logs/eval_compare/lm128-be10.log
# 23.61
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.15 0.15 300 1.0 24 > logs/eval_compare/lm128-be15.log
# 23.72
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.20 0.15 300 1.0 24 > logs/eval_compare/lm128-be20.log
# 24.04
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.30 0.15 300 1.0 24 > logs/eval_compare/lm128-be30.log
# 24.35
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.40 0.15 300 1.0 24 > logs/eval_compare/lm128-be40.log
# 25.14

## Language model Beam Width
# Based on the alpha and beta tests.
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 128 0.01 0.19 300 1.0 24 > logs/eval_compare/lm128.log
# 22.41
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1  256 0.01 0.19 300 1.0 24 > logs/eval_compare/lm256-2.log
# 20.76
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1  512 0.01 0.19 300 1.0 24 > logs/eval_compare/lm512-2.log
# 19.81
#./exp/paramtest21/rebel_eval_compare.sh "lm" "f" "t" 1 1024 0.01 0.19 300 1.0 24 > logs/eval_compare/lm1024-2.log
# 19.38



