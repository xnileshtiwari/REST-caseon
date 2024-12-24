<?php
header('Content-Type: application/json');

// Function to validate input
function validateInput($input) {
    return !empty(trim($input));
}

try {
    // Get JSON input
    $json = file_get_contents('php://input');
    $data = json_decode($json, true);

    // Check if it's a POST request
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
        throw new Exception('Only POST method is allowed');
    }

    // Get data from JSON
    $indexName = $data['index_name'] ?? '';
    $userInput = $data['user_input'] ?? '';

    // Validate inputs
    if (!validateInput($indexName) || !validateInput($userInput)) {
        throw new Exception('Index name and user input are required');
    }

    // Initialize cURL session
    $ch = curl_init('http://localhost:8000/chat');
    
    // Set cURL options
    curl_setopt($ch, CURLOPT_POST, 1);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode([
        'index_name' => $indexName,
        'user_input' => $userInput
    ]));
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, array(
        'Content-Type: application/json'
    ));

    // Execute cURL request
    $response = curl_exec($ch);
    
    // Check for cURL errors
    if (curl_errno($ch)) {
        throw new Exception('Curl error: ' . curl_error($ch));
    }
    
    // Close cURL session
    curl_close($ch);

    // Decode and send response
    $result = json_decode($response, true);
    echo json_encode([
        'status' => 'success',
        'data' => $result
    ]);

} catch (Exception $e) {
    http_response_code(500);
    echo json_encode([
        'status' => 'error',
        'message' => $e->getMessage()
    ]);
}
?>
