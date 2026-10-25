// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract ModelRegistry {

    struct ModelRecord {
        uint256 round;
        bytes32 modelHash;
        address submitter;
        uint256 timestamp;
    }

    ModelRecord[] private records;

    address public authorizedAggregator;

    constructor() {
        authorizedAggregator = msg.sender;
    }

    modifier onlyAuthorizedAggregator() {
        require(
            msg.sender == authorizedAggregator,
            "Not authorized"
        );
        _;
    }

    function storeModelHash(
        uint256 round,
        bytes32 modelHash
    )
        external
        onlyAuthorizedAggregator
    {
        records.push(
            ModelRecord({
                round: round,
                modelHash: modelHash,
                submitter: msg.sender,
                timestamp: block.timestamp
            })
        );
    }

    function getRecordCount()
        external
        view
        returns (uint256)
    {
        return records.length;
    }

    function getRecord(uint256 index)
        external
        view
        returns (
            uint256 round,
            bytes32 modelHash,
            address submitter,
            uint256 timestamp
        )
    {
        require(
            index < records.length,
            "Invalid index"
        );

        ModelRecord memory record = records[index];

        return (
            record.round,
            record.modelHash,
            record.submitter,
            record.timestamp
        );
    }

    function getAllRecords()
        external
        view
        returns (ModelRecord[] memory)
    {
        return records;
    }
}