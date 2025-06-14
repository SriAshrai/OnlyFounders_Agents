// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title PitchRegistry
 * @dev A smart contract to store verifiable pitch strength scores and their
 * corresponding Zero-Knowledge Proof (ZKP) hashes on-chain.
 * This acts as an immutable record of the Pitch Strength Agent's output.
 */
contract PitchRegistry {

    // Struct to define the data stored for each pitch analysis.
    // We store the overall score, component scores, and a hash referencing the ZKP.
    struct PitchScoreEntry {
        uint256 overallScore;            // Overall pitch score (e.g., 1-100)
        uint256 clarityScore;            // Clarity component score
        uint256 originalityScore;        // Originality component score
        uint256 teamStrengthScore;       // Team Strength component score
        uint256 marketFitScore;          // Market Fit component score
        bytes32 zkpHash;                 // Hash of the Zero-Knowledge Proof (e.g., hash of the public inputs/proof itself)
        uint256 timestamp;               // Timestamp of when the score was recorded
        address recordedBy;              // Address of the entity (e.g., the agent's smart contract or authorized EOA) that recorded the score
    }

    // Mapping from a unique pitch ID (bytes32) to its PitchScoreEntry.
    // This allows for efficient lookup of pitch scores.
    mapping(bytes32 => PitchScoreEntry) public pitchScores;

    // Event emitted when a new pitch score is recorded.
    // Useful for off-chain services to easily monitor new scores.
    event PitchScoreRecorded(
        bytes32 indexed pitchId,
        uint256 overallScore,
        bytes32 zkpHash,
        uint256 timestamp,
        address recordedBy
    );

    // Modifier to restrict access to functions, ensuring only authorized entities can record scores.
    // In a full system, this would likely be an Agent Management contract or a DAO.
    address public owner; // Simplistic owner for now, would be a more robust access control in production.

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    /**
     * @dev Records the pitch strength scores and ZKP hash on-chain.
     * This function should only be callable by the authorized Pitch Strength Agent.
     * @param _pitchId A unique identifier for the pitch. This could be a hash of the original pitch content,
     * or an ID agreed upon by the off-chain system.
     * @param _overallScore The aggregated overall score of the pitch.
     * @param _clarityScore Score for clarity of the pitch.
     * @param _originalityScore Score for originality of the pitch.
     * @param _teamStrengthScore Score for team strength as assessed by the agent.
     * @param _marketFitScore Score for market fit as assessed by the agent.
     * @param _zkpHash The cryptographic hash of the Zero-Knowledge Proof verifying the score calculation.
     * This hash would be verified off-chain by verifiers interested in the raw proof.
     */
    function recordPitchScore(
        bytes32 _pitchId,
        uint256 _overallScore,
        uint256 _clarityScore,
        uint256 _originalityScore,
        uint256 _teamStrengthScore,
        uint256 _marketFitScore,
        bytes32 _zkpHash
    ) external onlyOwner {
        require(pitchScores[_pitchId].timestamp == 0, "Pitch score already recorded for this ID");
        require(_overallScore > 0, "Overall score must be positive"); // Basic validation

        pitchScores[_pitchId] = PitchScoreEntry({
            overallScore: _overallScore,
            clarityScore: _clarityScore,
            originalityScore: _originalityScore,
            teamStrengthScore: _teamStrengthScore,
            marketFitScore: _marketFitScore,
            zkpHash: _zkpHash,
            timestamp: block.timestamp,
            recordedBy: msg.sender
        });

        emit PitchScoreRecorded(
            _pitchId,
            _overallScore,
            _zkpHash,
            block.timestamp,
            msg.sender
        );
    }

    /**
     * @dev Retrieves the PitchScoreEntry for a given pitch ID.
     * @param _pitchId The unique identifier of the pitch.
     * @return A tuple containing all stored score details and metadata.
     */
    function getPitchScore(bytes32 _pitchId)
        external
        view
        returns (
            uint256 overallScore,
            uint256 clarityScore,
            uint256 originalityScore,
            uint256 teamStrengthScore,
            uint256 marketFitScore,
            bytes32 zkpHash,
            uint256 timestamp,
            address recordedBy
        )
    {
        PitchScoreEntry storage entry = pitchScores[_pitchId];
        require(entry.timestamp != 0, "Pitch score not found for this ID");

        return (
            entry.overallScore,
            entry.clarityScore,
            entry.originalityScore,
            entry.teamStrengthScore,
            entry.marketFitScore,
            entry.zkpHash,
            entry.timestamp,
            entry.recordedBy
        );
    }
}

