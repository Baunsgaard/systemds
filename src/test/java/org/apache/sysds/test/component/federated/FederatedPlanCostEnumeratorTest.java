/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.component.federated;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class FederatedPlanCostEnumeratorTest extends AutomatedTestBase
{
	private static final String TEST_DIR = "functions/federated/privacy/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedPlanCostEnumeratorTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {}

	@Test
	public void testFederatedPlanCostEnumerator1() { runTest("FederatedPlanCostEnumeratorTest1.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator2() { runTest("FederatedPlanCostEnumeratorTest2.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator3() { runTest("FederatedPlanCostEnumeratorTest3.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator4() { runTest("FederatedPlanCostEnumeratorTest4.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator5() { runTest("FederatedPlanCostEnumeratorTest5.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator6() { runTest("FederatedPlanCostEnumeratorTest6.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator7() { runTest("FederatedPlanCostEnumeratorTest7.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator8() { runTest("FederatedPlanCostEnumeratorTest8.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator9() { runTest("FederatedPlanCostEnumeratorTest9.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator10() { runTest("FederatedPlanCostEnumeratorTest10.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator11() { runTest("FederatedPlanCostEnumeratorTest11.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator12() { runTest("FederatedPlanCostEnumeratorTest12.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator13() { runTest("FederatedPlanCostEnumeratorTest13.dml"); }

	private void runTest(String scriptFilename) {
		int index = scriptFilename.lastIndexOf(".dml");
		String testName = scriptFilename.substring(0, index > 0 ? index : scriptFilename.length());
		TestConfiguration testConfig = new TestConfiguration(TEST_CLASS_DIR, testName, new String[] {});
		addTestConfiguration(testName, testConfig);
		loadTestConfiguration(testConfig);
		
		try {
			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);
			
			// Set FEDERATED_PLANNER configuration to COMPILE_COST_BASED
			ConfigurationManager.getDMLConfig().setTextValue(DMLConfig.FEDERATED_PLANNER, "compile_cost_based");
			
			//read script
			String dmlScriptString = DMLScript.readDMLScript(true, HOME + scriptFilename);

			//parsing and dependency analysis
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, new HashMap<>());
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);
		}
		catch (Exception e) {
			e.printStackTrace();
			Assert.fail(e.getMessage());
		}
	}
}
