document.addEventListener('DOMContentLoaded', () => {
    const pitchFile = document.getElementById('pitchFile');
    const pitchText = document.getElementById('pitchText');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const buttonText = document.getElementById('buttonText');
    const resultsSection = document.getElementById('resultsSection');
    const messageBox = document.getElementById('messageBox');

    // Pitch Strength Agent display elements
    const psOverallScore = document.getElementById('psOverallScore');
    const psClarityScore = document.getElementById('psClarityScore');
    const psClarityReasoning = document.getElementById('psClarityReasoning');
    const psOriginalityScore = document.getElementById('psOriginalityScore');
    const psOriginalityReasoning = document.getElementById('psOriginalityReasoning');
    const psTeamStrengthScore = document.getElementById('psTeamStrengthScore');
    const psTeamStrengthReasoning = document.getElementById('psTeamStrengthReasoning');
    const psMarketFitScore = document.getElementById('psMarketFitScore');
    const psMarketFitReasoning = document.getElementById('psMarketFitReasoning');
    const psTeeProcessed = document.getElementById('psTeeProcessed');
    const psZkpHash = document.getElementById('psZkpHash');
    const psOnChainTx = document.getElementById('psOnChainTx');

    // Social Trust Graph Agent display elements
    const stgTrustScores = document.getElementById('stgTrustScores');
    const stgZkVerified = document.getElementById('stgZkVerified');
    const stgTeeAnalysis = document.getElementById('stgTeeAnalysis');
    const stgGraphHash = document.getElementById('stgGraphHash');
    const stgOnChainTx = document.getElementById('stgOnChainTx');

    // Overall Orchestration status
    const overallStatus = document.getElementById('overallStatus');

    let sessionCounter = 0; // Simple counter for unique session IDs

    // Define your backend API endpoint
    // IMPORTANT: This URL MUST match the host and port where your Flask app.py is running.
    const BACKEND_API_URL = 'http://localhost:5000/analyze-agents'; // Matches your Flask app.py route

    function showMessage(message, type = 'info') {
        messageBox.textContent = message;
        messageBox.className = `p-4 rounded-lg text-lg mb-4 block ${type === 'error' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`;
        messageBox.classList.remove('hidden');
    }

    function hideMessage() {
        messageBox.classList.add('hidden');
    }

    function setLoading(isLoading) {
        if (isLoading) {
            buttonText.textContent = 'Running Orchestration...';
            loadingSpinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
            clearBtn.disabled = true;
            hideMessage();
            resultsSection.classList.add('hidden');
            overallStatus.textContent = 'Processing...';
        } else {
            buttonText.textContent = 'Run Full Orchestration';
            loadingSpinner.classList.add('hidden');
            analyzeBtn.disabled = false;
            clearBtn.disabled = false;
        }
    }

    analyzeBtn.addEventListener('click', async () => {
        setLoading(true);
        let pitchContent = pitchText.value.trim();
        let file = pitchFile.files[0];
        let inputType = '';
        let fileText = '';

        if (file) {
            // For this demo, we'll convert file content to text and send it directly.
            // In a real application, you might handle binary files differently (e.g., base64 encode or upload to cloud storage).
            try {
                fileText = await readFileAsync(file);
                if (fileText.length > 20000) { // Keep file size reasonable for browser processing
                    showMessage("File content is too large. Please keep it under 20KB for this demo.", "error");
                    setLoading(false);
                    return;
                }
                pitchContent = fileText; // Use file content as pitch_input_content
                inputType = 'file'; // Just for tracking input type
            } catch (error) {
                showMessage(`Error reading file: ${error.message}`, "error");
                setLoading(false);
                return;
            }
        } else if (pitchContent) {
            inputType = 'text';
            if (pitchContent.length > 10000) { // Keep text length reasonable
                showMessage("Pitch text is too long. Please keep it under 10KB for this demo.", "error");
                setLoading(false);
                return;
            }
        } else {
            showMessage("Please upload a pitch file or paste pitch text.", "error");
            setLoading(false);
            return;
        }

        sessionCounter++;
        const currentSessionId = `session_${String(sessionCounter).padStart(3, '0')}`;

        // Payload sent to the Flask backend
        // We send the extracted text content. The backend will use this directly.
        const payload = {
            user_session_id: currentSessionId,
            pitch_input_content: pitchContent,
            // We no longer send file_path from frontend, as backend processes content directly
        };

        try {
            console.log("Sending payload to backend:", payload);
            const response = await fetch(BACKEND_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json', // Crucial: tells the backend the body is JSON
                },
                body: JSON.stringify(payload), // Convert JS object to JSON string for network transmission
            });

            // Check if the HTTP response itself was successful (e.g., 200 OK)
            if (!response.ok) {
                // If response status is not 2xx, it's an HTTP error
                const errorData = await response.json(); // Try to parse error message from backend
                showMessage(`Backend HTTP Error (${response.status}): ${errorData.error || 'Unknown error'}`, "error");
                overallStatus.textContent = 'Failed';
                resultsSection.classList.add('hidden');
                return; // Stop further processing
            }

            const results = await response.json(); // Parse the JSON response from the backend

            // Now check the 'status' field within the JSON payload from the backend
            if (results.status === 'failed') {
                showMessage(`Orchestration Failed: ${results.error || 'Unknown error from agents'}`, "error");
                overallStatus.textContent = 'Failed';
                resultsSection.classList.add('hidden');
            } else {
                displayResults(results);
                showMessage("Multi-Agent Orchestration completed successfully!", "success");
                overallStatus.textContent = 'Completed';
                resultsSection.classList.remove('hidden');
            }

        } catch (error) {
            console.error("Error during orchestration (network or parsing):", error);
            showMessage(`A network or parsing error occurred: ${error.message}. Make sure your Python backend is running at ${BACKEND_API_URL}.`, "error");
            overallStatus.textContent = 'Failed';
            resultsSection.classList.add('hidden');
        } finally {
            setLoading(false);
        }
    });

    clearBtn.addEventListener('click', () => {
        pitchFile.value = '';
        pitchText.value = '';
        resultsSection.classList.add('hidden');
        hideMessage();
        setLoading(false);
        overallStatus.textContent = '';
    });

    // Helper function to read file content as text
    function readFileAsync(file) {
        return new Promise((resolve, reject) => {
            let reader = new FileReader();
            reader.onload = () => {
                resolve(reader.result);
            };
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    // Function to display results in the UI (this part remains largely the same)
    function displayResults(results) {
        const psResults = results.pitch_analysis;
        const stgResults = results.social_trust_analysis;

        // Display Pitch Strength Agent Results
        if (psResults) {
            psOverallScore.textContent = psResults.overall_score || 'N/A';
            psClarityScore.textContent = psResults.component_scores.clarity || 'N/A';
            psClarityReasoning.textContent = psResults.analysis_results.components.clarity?.reasoning || '';
            psOriginalityScore.textContent = psResults.component_scores.originality || 'N/A';
            psOriginalityReasoning.textContent = psResults.analysis_results.components.originality?.reasoning || '';
            psTeamStrengthScore.textContent = psResults.component_scores.team_strength || 'N/A';
            psTeamStrengthReasoning.textContent = psResults.analysis_results.components.team_strength?.reasoning || '';
            psMarketFitScore.textContent = psResults.component_scores.market_fit || 'N/A';
            psMarketFitReasoning.textContent = psResults.analysis_results.components.market_fit?.reasoning || '';
            psTeeProcessed.textContent = psResults.privacy_flags.tee_processed ? 'Yes' : 'No';
            psZkpHash.textContent = psResults.privacy_flags.zkp_hash || 'N/A';
            psOnChainTx.textContent = psResults.on_chain_tx_hash || 'N/A';
        } else {
            psOverallScore.textContent = 'N/A';
            psClarityScore.textContent = 'N/A';
            psClarityReasoning.textContent = '';
            psOriginalityScore.textContent = 'N/A';
            psOriginalityReasoning.textContent = '';
            psTeamStrengthScore.textContent = 'N/A';
            psTeamStrengthReasoning.textContent = '';
            psMarketFitScore.textContent = 'N/A';
            psMarketFitReasoning.textContent = '';
            psTeeProcessed.textContent = 'N/A';
            psZkpHash.textContent = 'N/A';
            psOnChainTx.textContent = 'N/A';
        }

        // Display Social Trust Graph Agent Results
        if (stgResults) {
            stgTrustScores.innerHTML = ''; // Clear previous scores
            if (stgResults.overall_trust_scores && Object.keys(stgResults.overall_trust_scores).length > 0) {
                const sortedScores = Object.entries(stgResults.overall_trust_scores)
                                             .sort(([,scoreA], [,scoreB]) => scoreB - scoreA)
                                             .slice(0, 5); // Display top 5 entities
                sortedScores.forEach(([entity, score]) => {
                    const p = document.createElement('p');
                    p.innerHTML = `<span class="font-medium text-gray-700">${entity}:</span> <span class="text-blue-500">${score}</span>`;
                    stgTrustScores.appendChild(p);
                });
            } else {
                stgTrustScores.textContent = 'No trust scores calculated or available.';
            }

            stgZkVerified.textContent = stgResults.privacy_flags.zk_verified_endorsements_processed ? 'Yes' : 'No';
            stgTeeAnalysis.textContent = stgResults.privacy_flags.tee_analysis_conducted ? 'Yes' : 'No';
            stgGraphHash.textContent = stgResults.current_graph_hash || 'N/A';
            stgOnChainTx.textContent = stgResults.on_chain_tx_hash || 'N/A';
        } else {
            stgTrustScores.textContent = 'No social trust analysis results.';
            stgZkVerified.textContent = 'N/A';
            stgTeeAnalysis.textContent = 'N/A';
            stgGraphHash.textContent = 'N/A';
            stgOnChainTx.textContent = 'N/A';
        }

        overallStatus.textContent = results.status;
    }
});

    