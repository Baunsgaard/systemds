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

package org.apache.sysml.test.integration.functions.misc;

import org.junit.Test;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

public class InvalidFunctionAssignmentTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_NAME1 = "InvalidFunctionAssignmentTest1";
	private final static String TEST_NAME2 = "InvalidFunctionAssignmentTest2";
	private final static String TEST_NAME3 = "InvalidFunctionAssignmentTest3";
	private final static String TEST_NAME4 = "InvalidFunctionAssignmentTest4";
	private final static String TEST_CLASS_DIR = TEST_DIR + InvalidFunctionAssignmentTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {}));
	}
	
	@Test
	public void testValidFunctionAssignmentInlined() {
		runTest( TEST_NAME1, false );
	}
	
	@Test
	public void testInvalidFunctionAssignmentInlined() {
		//MB: we now support UDFs in expressions and hence it's valid
		runTest( TEST_NAME2, false );
	}
	
	@Test
	public void testValidFunctionAssignment() {
		runTest( TEST_NAME3, false );
	}
	
	@Test
	public void testInvalidFunctionAssignment() {
		//MB: we now support UDFs in expressions and hence it's valid
		runTest( TEST_NAME4, false );
	}
	
	private void runTest( String testName, boolean exceptionExpected ) {
		TestConfiguration config = getTestConfiguration(testName);
		loadTestConfiguration(config);
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{};
		runTest(true, exceptionExpected, DMLException.class, -1);
	}
}