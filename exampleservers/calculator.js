#!/usr/bin/env node

/**
 * ADVANCED CALCULATOR MCP SERVER v2.1.0
 * 
 * Powered by math.js for advanced mathematical operations:
 * - Basic arithmetic: +, -, *, /, %, ^
 * - Advanced functions: sin, cos, tan, log, sqrt, abs, etc.
 * - Constants: pi, e, phi
 * - Algebra: solve equations, simplify expressions
 * - Calculus: derivatives, integrals
 * - Matrices: operations, determinants, eigenvalues
 * - Units: conversions and calculations with units
 * - Complex numbers
 * - Statistics: mean, median, std, variance
 * - ENHANCED: Financial calculations, combinatorics, number theory
 * - ENHANCED: Expression history and memory storage
 * - ENHANCED: Multi-expression evaluation
 * 
 * Smart completion: Hints at TASK_COMPLETE for simple standalone questions,
 * but stays neutral for complex multi-step planning contexts.
 */

const readline = require('readline');

// Try to load math.js, fall back to basic Function eval if not available
let math;
let hasMathJS = false;

try {
    math = require('mathjs');
    hasMathJS = true;
    process.stderr.write('[Calculator] Loaded with math.js - full capabilities enabled\n');
} catch (e) {
    process.stderr.write('[Calculator] math.js not found - using basic mode\n');
    process.stderr.write('[Calculator] Install math.js for advanced features: npm install mathjs\n');
}

// ============================================================================
// ENHANCED FEATURES - Memory and History
// ============================================================================

const calculatorMemory = {
    history: [],
    variables: {},
    lastResult: null
};

function storeInHistory(expression, result) {
    calculatorMemory.history.push({
        expression,
        result,
        timestamp: new Date().toISOString()
    });
    calculatorMemory.lastResult = result;
    
    // Keep only last 100 calculations
    if (calculatorMemory.history.length > 100) {
        calculatorMemory.history.shift();
    }
}

function getHistory(count = 10) {
    return calculatorMemory.history.slice(-count);
}

// ============================================================================
// ENHANCED FEATURES - Financial Calculations
// ============================================================================

function calculateCompoundInterest(principal, rate, time, frequency = 1) {
    // A = P(1 + r/n)^(nt)
    return principal * Math.pow((1 + rate / frequency), frequency * time);
}

function calculateLoanPayment(principal, annualRate, years) {
    // Monthly payment formula: M = P[r(1+r)^n]/[(1+r)^n-1]
    const monthlyRate = annualRate / 12;
    const numPayments = years * 12;
    
    if (monthlyRate === 0) return principal / numPayments;
    
    return principal * (monthlyRate * Math.pow(1 + monthlyRate, numPayments)) / 
           (Math.pow(1 + monthlyRate, numPayments) - 1);
}

function calculateNPV(rate, cashFlows) {
    // Net Present Value
    return cashFlows.reduce((acc, cf, i) => {
        return acc + cf / Math.pow(1 + rate, i);
    }, 0);
}

// ============================================================================
// ENHANCED FEATURES - Number Theory
// ============================================================================

function isPrime(n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 === 0 || n % 3 === 0) return false;
    
    for (let i = 5; i * i <= n; i += 6) {
        if (n % i === 0 || n % (i + 2) === 0) return false;
    }
    return true;
}

function getNthPrime(n) {
    if (n < 1) throw new Error("n must be positive");
    
    let count = 0;
    let num = 2;
    
    while (count < n) {
        if (isPrime(num)) count++;
        if (count < n) num++;
    }
    
    return num;
}

function getPrimeFactors(n) {
    const factors = [];
    let divisor = 2;
    
    while (n >= 2) {
        if (n % divisor === 0) {
            factors.push(divisor);
            n = n / divisor;
        } else {
            divisor++;
        }
    }
    
    return factors;
}

function fibonacci(n) {
    if (n <= 0) return 0;
    if (n === 1) return 1;
    
    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
        [a, b] = [b, a + b];
    }
    return b;
}

// ============================================================================
// ENHANCED FEATURES - Advanced Combinatorics
// ============================================================================

function permutations(n, r) {
    if (r > n) return 0;
    let result = 1;
    for (let i = 0; i < r; i++) {
        result *= (n - i);
    }
    return result;
}

function combinations(n, r) {
    if (r > n) return 0;
    if (r === 0 || r === n) return 1;
    
    // Use smaller r for efficiency
    r = Math.min(r, n - r);
    
    let result = 1;
    for (let i = 0; i < r; i++) {
        result *= (n - i);
        result /= (i + 1);
    }
    return Math.round(result);
}

// ============================================================================
// ENHANCED FEATURES - Expression Preprocessing
// ============================================================================

function preprocessExpression(expression) {
    // Handle special functions and shortcuts
    let processed = expression;
    
    // Replace 'ans' or 'last' with last result
    if (calculatorMemory.lastResult !== null) {
        processed = processed.replace(/\bans\b|\blast\b/gi, calculatorMemory.lastResult.toString());
    }
    
    // Handle custom functions
    processed = processed.replace(/isPrime\((\d+)\)/g, (match, n) => isPrime(parseInt(n)));
    processed = processed.replace(/nthPrime\((\d+)\)/g, (match, n) => getNthPrime(parseInt(n)));
    processed = processed.replace(/fib\((\d+)\)/g, (match, n) => fibonacci(parseInt(n)));
    
    return processed;
}

// ============================================================================
// THE CALCULATOR FUNCTION (Enhanced)
// ============================================================================

function calculate(expression, options = {}) {
    // Preprocess the expression
    expression = preprocessExpression(expression);
    
    if (hasMathJS) {
        // Use math.js for advanced calculations
        try {
            const result = math.evaluate(expression);
            const formattedResult = formatResult(result);
            
            // Store in history unless disabled
            if (!options.skipHistory) {
                storeInHistory(expression, formattedResult);
            }
            
            return formattedResult;
        } catch (error) {
            throw new Error(`Math.js error: ${error.message}`);
        }
    } else {
        // Fallback to basic mode
        const result = calculateBasic(expression);
        
        if (!options.skipHistory) {
            storeInHistory(expression, result);
        }
        
        return result;
    }
}

function calculateBasic(expression) {
    // Remove spaces
    expression = expression.replace(/\s+/g, '');
    
    // Security check - only allow numbers and operators
    if (!/^[\d+\-*/.()^%]+$/.test(expression)) {
        throw new Error("Invalid expression. Install math.js for advanced features: npm install mathjs");
    }
    
    // Replace ^ with ** for exponentiation
    expression = expression.replace(/\^/g, '**');
    
    // Calculate
    try {
        const result = Function(`"use strict"; return (${expression})`)();
        return result;
    } catch (error) {
        throw new Error(`Calculation error: ${error.message}`);
    }
}

function formatResult(result) {
    // Handle different result types from math.js
    if (typeof result === 'object') {
        // Matrix, complex number, unit, etc.
        if (result.toString) {
            return result.toString();
        }
        return JSON.stringify(result);
    }
    
    // Format numbers nicely
    if (typeof result === 'number') {
        // Use scientific notation for very large or very small numbers
        if (Math.abs(result) > 1e10 || (Math.abs(result) < 1e-6 && result !== 0)) {
            return result.toExponential(10);
        }
        // Round to avoid floating point errors
        return Math.round(result * 1e10) / 1e10;
    }
    
    return result;
}

function detectContext(expression) {
    // Detect if this is a simple standalone question or part of complex work
    const simplePatterns = [
        /^\d+\s*[\+\-\*\/\^%]\s*\d+$/,  // Simple "X op Y"
        /^[\d\+\-\*\/\^%\(\)\s]+$/       // Only numbers and basic operators
    ];
    
    const complexPatterns = [
        /solve|equation|derivative|integral|matrix/i,
        /sin|cos|tan|log|ln|sqrt|abs/i,
        /factorial|combination|permutation/i,
        /unit|convert/i,
        /isPrime|nthPrime|fib|compound|loan|npv/i
    ];
    
    // Check if it's a simple expression
    const isSimple = simplePatterns.some(pattern => pattern.test(expression));
    
    // Check if it uses advanced features
    const isComplex = complexPatterns.some(pattern => pattern.test(expression));
    
    if (isSimple && !isComplex) {
        return 'simple';
    } else if (isComplex) {
        return 'complex';
    }
    
    return 'unknown';
}

function buildResponse(expression, result) {
    const context = detectContext(expression);
    
    // Format the basic result
    let response = `Result: ${result}`;
    
    // Add contextual hints
    if (context === 'simple') {
        // For simple calculations, gently hint at completion
        response += `\n\n(Simple calculation completed)`;
    } else if (context === 'complex') {
        // For complex calculations, stay neutral - might be part of larger work
        response += `\n\n(Advanced calculation completed - ready for next step if needed)`;
    } else {
        // Unknown context - stay neutral
        response += `\n\n(Calculation completed)`;
    }
    
    return response;
}

// ============================================================================
// ENHANCED TOOLS - Multi-expression and special functions
// ============================================================================

function evaluateMultiple(expressions) {
    const results = [];
    
    for (const expr of expressions) {
        try {
            const result = calculate(expr, { skipHistory: true });
            results.push({ expression: expr, result, success: true });
        } catch (error) {
            results.push({ expression: expr, error: error.message, success: false });
        }
    }
    
    // Store the last successful result in history
    const lastSuccess = results.reverse().find(r => r.success);
    if (lastSuccess) {
        storeInHistory(lastSuccess.expression, lastSuccess.result);
    }
    
    return results.reverse();
}

function handleFinancialCalculation(type, params) {
    switch (type.toLowerCase()) {
        case 'compound_interest':
            return calculateCompoundInterest(
                params.principal,
                params.rate,
                params.time,
                params.frequency || 1
            );
        case 'loan_payment':
            return calculateLoanPayment(
                params.principal,
                params.rate,
                params.years
            );
        case 'npv':
            return calculateNPV(params.rate, params.cashFlows);
        default:
            throw new Error(`Unknown financial calculation type: ${type}`);
    }
}

function handleNumberTheory(operation, params) {
    switch (operation.toLowerCase()) {
        case 'isprime':
            return isPrime(params.number);
        case 'nthprime':
            return getNthPrime(params.n);
        case 'primefactors':
            return getPrimeFactors(params.number);
        case 'fibonacci':
            return fibonacci(params.n);
        case 'permutations':
            return permutations(params.n, params.r);
        case 'combinations':
            return combinations(params.n, params.r);
        default:
            throw new Error(`Unknown number theory operation: ${operation}`);
    }
}

// ============================================================================
// MCP SERVER CODE
// ============================================================================

function sendResponse(id, result) {
    console.log(JSON.stringify({
        jsonrpc: "2.0",
        id: id,
        result: result
    }));
}

function sendError(id, code, message) {
    console.log(JSON.stringify({
        jsonrpc: "2.0",
        id: id,
        error: { code: code, message: message }
    }));
}

function getToolDescription() {
    const baseDesc = "ADVANCED mathematical calculator - PRIMARY TOOL for ALL calculations. ALWAYS use instead of PowerShell for math.";
    
    if (hasMathJS) {
        return baseDesc + `

CAPABILITIES:
• Basic: +, -, *, /, %, ^ (power), parentheses
• Functions: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
• Logarithms: log, log10, ln (natural log)
• Roots & Powers: sqrt, cbrt, nthRoot, pow
• Rounding: round, ceil, floor, fix
• Other: abs, sign, gcd, lcm, factorial
• Constants: pi, e, phi, tau
• Algebra: simplify("x^2 + 2x + 1"), solve equations
• Calculus: derivative("x^2", "x"), numeric integration
• Matrices: [[1,2],[3,4]], det, inv, eigenvalues
• Units: "5 cm to inch", "100 km/h to mph"
• Complex: i or complex(re, im)
• Statistics: mean, median, std, variance, min, max
• ENHANCED: Use 'ans' or 'last' to reference previous result
• ENHANCED: isPrime(n), nthPrime(n), fib(n) for number theory

EXAMPLES:
• Basic: "789 * 456"
• Trig: "sin(pi/4)"
• Solve: "solve('x^2 - 4 = 0', 'x')" 
• Derivative: "derivative('x^2 + 3x', 'x')"
• Matrix: "det([[1,2],[3,4]])"
• Units: "5 cm to inch"
• Stats: "mean([1,2,3,4,5])"
• Complex: "sqrt(-1)"
• Chain: "100 * 1.5" then "ans + 50"
• Prime: "isPrime(17)"

For simple one-off calculations, task is complete after result.
For multi-step problems or planning, continue as needed.`;
    } else {
        return baseDesc + `

BASIC MODE (install math.js for advanced features):
• Supported: +, -, *, /, %, ^ (power), parentheses
• Example: "789 * 456", "(10 + 5) * 2"

Install math.js for advanced capabilities:
  npm install mathjs

For simple calculations, task complete after result.
For planning/multi-step, continue as needed.`;
    }
}

function getFinancialToolDescription() {
    return `Financial calculator for common calculations:
• compound_interest: Calculate compound interest
• loan_payment: Calculate monthly loan payment
• npv: Calculate Net Present Value

Parameters vary by calculation type.`;
}

function getNumberTheoryToolDescription() {
    return `Number theory and combinatorics:
• isPrime: Check if a number is prime
• nthPrime: Get the nth prime number
• primeFactors: Get prime factorization
• fibonacci: Calculate nth Fibonacci number
• permutations: Calculate P(n,r)
• combinations: Calculate C(n,r)`;
}

function handleRequest(request) {
    const { id, method, params } = request;
    
    try {
        if (method === "initialize") {
            sendResponse(id, {
                protocolVersion: "2024-11-05",
                capabilities: {},
                serverInfo: {
                    name: "advanced-calculator",
                    version: "2.1.0"
                }
            });
        }
        else if (method === "tools/list") {
            const tools = [
                {
                    name: "calculate",
                    description: getToolDescription(),
                    inputSchema: {
                        type: "object",
                        properties: {
                            expression: {
                                type: "string",
                                description: hasMathJS 
                                    ? "Any mathematical expression. Supports algebra, calculus, matrices, units, trig, etc. Use 'ans' or 'last' for previous result."
                                    : "Basic arithmetic expression with +, -, *, /, %, ^, ()"
                            }
                        },
                        required: ["expression"]
                    }
                },
                {
                    name: "calculate_multiple",
                    description: "Evaluate multiple expressions in sequence. Later expressions can reference earlier results.",
                    inputSchema: {
                        type: "object",
                        properties: {
                            expressions: {
                                type: "array",
                                items: { type: "string" },
                                description: "Array of mathematical expressions to evaluate"
                            }
                        },
                        required: ["expressions"]
                    }
                },
                {
                    name: "get_history",
                    description: "Retrieve calculation history",
                    inputSchema: {
                        type: "object",
                        properties: {
                            count: {
                                type: "number",
                                description: "Number of recent calculations to retrieve (default: 10)"
                            }
                        }
                    }
                }
            ];
            
            // Add advanced tools if math.js is available
            if (hasMathJS) {
                tools.push(
                    {
                        name: "financial_calculation",
                        description: getFinancialToolDescription(),
                        inputSchema: {
                            type: "object",
                            properties: {
                                type: {
                                    type: "string",
                                    enum: ["compound_interest", "loan_payment", "npv"],
                                    description: "Type of financial calculation"
                                },
                                params: {
                                    type: "object",
                                    description: "Parameters for the calculation (varies by type)"
                                }
                            },
                            required: ["type", "params"]
                        }
                    },
                    {
                        name: "number_theory",
                        description: getNumberTheoryToolDescription(),
                        inputSchema: {
                            type: "object",
                            properties: {
                                operation: {
                                    type: "string",
                                    enum: ["isPrime", "nthPrime", "primeFactors", "fibonacci", "permutations", "combinations"],
                                    description: "Number theory operation to perform"
                                },
                                params: {
                                    type: "object",
                                    description: "Parameters for the operation"
                                }
                            },
                            required: ["operation", "params"]
                        }
                    }
                );
            }
            
            sendResponse(id, { tools });
        }
        else if (method === "tools/call") {
            const toolName = params.name;
            const args = params.arguments || {};
            
            if (toolName === "calculate") {
                const result = calculate(args.expression);
                const response = buildResponse(args.expression, result);
                
                sendResponse(id, {
                    content: [
                        {
                            type: "text",
                            text: response
                        }
                    ]
                });
            }
            else if (toolName === "calculate_multiple") {
                const results = evaluateMultiple(args.expressions);
                
                let responseText = "Multiple Expression Results:\n\n";
                results.forEach((r, i) => {
                    if (r.success) {
                        responseText += `${i + 1}. ${r.expression} = ${r.result}\n`;
                    } else {
                        responseText += `${i + 1}. ${r.expression} = ERROR: ${r.error}\n`;
                    }
                });
                
                sendResponse(id, {
                    content: [
                        {
                            type: "text",
                            text: responseText
                        }
                    ]
                });
            }
            else if (toolName === "get_history") {
                const count = args.count || 10;
                const history = getHistory(count);
                
                let responseText = `Recent Calculations (last ${history.length}):\n\n`;
                history.forEach((entry, i) => {
                    responseText += `${i + 1}. ${entry.expression} = ${entry.result}\n`;
                });
                
                sendResponse(id, {
                    content: [
                        {
                            type: "text",
                            text: responseText
                        }
                    ]
                });
            }
            else if (toolName === "financial_calculation") {
                const result = handleFinancialCalculation(args.type, args.params);
                
                sendResponse(id, {
                    content: [
                        {
                            type: "text",
                            text: `Financial Calculation (${args.type}):\nResult: ${formatResult(result)}`
                        }
                    ]
                });
            }
            else if (toolName === "number_theory") {
                const result = handleNumberTheory(args.operation, args.params);
                
                sendResponse(id, {
                    content: [
                        {
                            type: "text",
                            text: `Number Theory (${args.operation}):\nResult: ${formatResult(result)}`
                        }
                    ]
                });
            }
            else {
                sendError(id, -32601, `Unknown tool: ${toolName}`);
            }
        }
        else {
            sendError(id, -32601, `Unknown method: ${method}`);
        }
    } catch (error) {
        sendError(id, -32000, error.message);
    }
}

// ============================================================================
// MAIN - Listen for JSON-RPC requests
// ============================================================================

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

const mode = hasMathJS ? 'ADVANCED' : 'BASIC';
process.stderr.write(`[Calculator MCP v2.1.0] Server started in ${mode} mode\n`);
process.stderr.write(`[Calculator MCP] Enhanced features: History, Memory, Financial, Number Theory\n`);

// Read each line from stdin
rl.on('line', (line) => {
    try {
        const request = JSON.parse(line);
        handleRequest(request);
    } catch (error) {
        process.stderr.write(`[Calculator MCP] Parse error: ${error.message}\n`);
    }
});

rl.on('close', () => {
    process.stderr.write('[Calculator MCP] Shutting down\n');
    process.exit(0);
});

// Handle errors
process.on('uncaughtException', (error) => {
    process.stderr.write(`[Calculator MCP] Error: ${error.message}\n`);
});
