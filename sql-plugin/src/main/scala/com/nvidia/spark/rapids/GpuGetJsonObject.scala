/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import scala.util.parsing.combinator.RegexParsers

import ai.rapids.cudf.ColumnVector
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.JSONUtils

import org.apache.spark.sql.catalyst.expressions.{ExpectsInputTypes, Expression}
import org.apache.spark.sql.rapids.test.CpuGetJsonObject
import org.apache.spark.sql.types.{DataType, StringType}

case class GpuGetJsonObject(json: Expression, path: Expression, verifyBetweenCpuAndGpu: Boolean,
    verifyDiffSavePath: String)
    extends GpuBinaryExpressionArgsAnyScalar
        with ExpectsInputTypes {
  override def left: Expression = json
  override def right: Expression = path
  override def dataType: DataType = StringType
  override def inputTypes: Seq[DataType] = Seq(StringType, StringType)
  override def nullable: Boolean = true
  override def prettyName: String = "get_json_object"

  private var cachedInstructions: 
      Option[Option[List[PathInstruction]]] = None

  def parseJsonPath(path: GpuScalar): Option[List[PathInstruction]] = {
    if (path.isValid) {
      val pathStr = path.getValue.toString()
      JsonPathParser.parse(pathStr)
    } else {
      None
    }
  }

  override def doColumnar(lhs: GpuColumnVector, rhs: GpuScalar): ColumnVector = {
    val fromGpu = lhs.getBase().getJSONObject(rhs.getBase)

    // Below is only for testing purpose
    if (verifyBetweenCpuAndGpu) { // enable verify diff between Cpu and Gpu
      withResource(CpuGetJsonObject.getJsonObjectOnCpu(lhs, rhs)) { fromCpu =>
        // verify result, save diffs if has
        CpuGetJsonObject.verify(lhs.getBase, fromGpu, fromCpu, verifyDiffSavePath)
      }
    }

    fromGpu
  }

  override def doColumnar(numRows: Int, lhs: GpuScalar, rhs: GpuScalar): ColumnVector = {
    withResource(GpuColumnVector.from(lhs, numRows, left.dataType)) { expandedLhs =>
      doColumnar(expandedLhs, rhs)
    }
  }
}
