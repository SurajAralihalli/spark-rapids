# Copyright (c) 2022-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import difflib
import sys

from data_gen import *
from spark_session import *

if not is_jvm_charset_utf8():
    pytestmark = [pytest.mark.regexp_perf, pytest.mark.skip(reason=str("Current locale doesn't support UTF-8, regexp support is disabled"))]
else:
    pytestmark = pytest.mark.regexp_perf


def mk_str_gen(pattern):
    return StringGen(pattern).with_special_case('').with_special_pattern('.{0,10}')

def do_cudf_rlike_test(spark, name, str_gen, num_regexes=10):
    re_gen = StringGen('[bf]o{0,2}:?\\+?\\$')
    df = unary_op_df(spark, str_gen)
    regexes = gen_scalar_values(re_gen, num_regexes, force_no_nulls=True)
    exprs = ["a"] + [f"(a rlike '{regex}')" for regex in regexes]
    transpiled = df.selectExpr(*exprs).collect()
    spark.conf.set("spark.rapids.sql.regexp.transpiler.enabled", False)
    df = unary_op_df(spark, str_gen)
    cudf = df.selectExpr(*exprs).collect()
    print(name)
    sys.stdout.writelines(difflib.unified_diff(
        a=[f"{x}\n" for x in transpiled],
        b=[f"{x}\n" for x in cudf],
        fromfile='TRANSPILED OUTPUT',
        tofile='CUDF OUTPUT'))


def do_cudf_extract_test(spark, name, str_gen, transpile, num_regexes=1):
    re_gen = StringGen('\\([bf]oo:?\\+?\\)\\$')
    # df = unary_op_df(spark, str_gen)
    # regexes = gen_scalar_values(re_gen, num_regexes, force_no_nulls=True)
    regexes = ['(boo:+)$']
    exprs = ["a"] + [f"regexp_extract(a,'{regex}', 1)" for regex in regexes]
    # transpiled = df.selectExpr(*exprs).collect()
    spark.conf.set("spark.rapids.sql.regexp.transpiler.enabled", transpile)
    df = unary_op_df(spark, str_gen)
    # cudf = df.selectExpr(*exprs).collect()
    print(name)
    debug_df(df.selectExpr(*exprs))
    # sys.stdout.writelines(difflib.unified_diff(
    #     a=[f"{x}\n" for x in transpiled],
    #     b=[f"{x}\n" for x in cudf],
    #     fromfile='TRANSPILED OUTPUT',
    #     tofile='CUDF OUTPUT'))


def test_re_rlike_newline(request):
    str_gen = mk_str_gen('([bf]o{0,2}|:){1,100}\n') \
        .with_special_case('boo:and:foo\n')
    with_gpu_session(lambda spark: do_cudf_rlike_test(spark, request.node.name, str_gen))
    

def test_re_rlike_line_terminators(request):
    str_gen = mk_str_gen('([bf]o{0,2}|:){1,100}(\r\n)|[\r\n\u0085\u2028\u2029]') \
        .with_special_case('boo:and:foo\n') \
        .with_special_case('boo:and:foo\r\n')
    with_gpu_session(lambda spark: do_cudf_rlike_test(spark, request.node.name, str_gen))

@pytest.mark.parametrize('transpile', [True, False], ids=idfn)
def test_re_extract_newline(request, transpile):
    str_gen = mk_str_gen('([bf]oo|:){1,100}\n') \
        .with_special_case('boo:and:foo\n')
    with_gpu_session(lambda spark: do_cudf_extract_test(spark, request.node.name, str_gen, transpile))

@pytest.mark.parametrize('transpile', [True, False], ids=idfn)
def test_re_extract_line_terminators(request, transpile):
    str_gen = mk_str_gen('([bf]oo|:){1,100}(\r\n)|[\r\n\u0085\u2028\u2029]') \
        .with_special_case('boo:and:foo\n') \
        .with_special_case('boo:and:foo\r\n')
    with_gpu_session(lambda spark: do_cudf_extract_test(spark, request.node.name, str_gen, transpile))



