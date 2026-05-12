#pragma once

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <fstream>
#include <map>
#include <algorithm>
#include <cmath>

struct OrderBookEntrySeq {
    uint32_t stockID;
    bool orderType; 
    uint8_t orderQty;
    uint8_t orderValue;
};

void updateDisplay_seq(const std::vector<uint64_t> &orderBook, int32_t freq);
int64_t totalAmountTraded_seq(const std::vector<uint64_t> &orderBook);
void printOrderStats_seq(const std::vector<uint64_t> &orderBook);
uint64_t unstuffBitsSeq(uint64_t packet);
OrderBookEntrySeq decodePacketSeq(uint64_t packet);