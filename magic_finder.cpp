#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm> // For std::min
#include <iomanip>   // For std::hex, std::setw, std::setfill
#include <stdexcept> // For std::runtime_error

#include "types.h" // For Bitboard, BoardLocation (if not directly defined here)
#include "magic_utils.h" // For common magic utilities


// Random number generator for magic candidates
std::mt19937_64 rng_magic(std::random_device{}()); // Keep this local to finder

chaturaji_cpp::Bitboard generate_random_magic_candidate() { // Keep this local to finder
    // "Sparse" magics (fewer bits set) often work better/faster
    return rng_magic() & rng_magic() & rng_magic();
}

// Function to find a magic number for a given square and piece type
chaturaji_cpp::Bitboard find_magic_for_square(int sq, bool is_rook,
                                 std::vector<chaturaji_cpp::Bitboard>& occupancies, // Out param
                                 std::vector<chaturaji_cpp::Bitboard>& attacks,     // Out param
                                 int& shift_bits)                       // Out param
{
    // Use functions from magic_utils namespace
    using namespace chaturaji_cpp::magic_utils;

    Bitboard blocker_mask = is_rook ? generate_rook_mask(sq) : generate_bishop_mask(sq);
    int num_mask_bits = pop_count(blocker_mask);
    shift_bits = 64 - num_mask_bits; // This is the common shift

    size_t num_occupancy_permutations = 1ULL << num_mask_bits;
    occupancies.assign(num_occupancy_permutations, 0ULL);
    attacks.assign(num_occupancy_permutations, 0ULL);

    // 1. Generate all occupancy permutations and their corresponding true attack sets
    for (size_t i = 0; i < num_occupancy_permutations; ++i) {
        occupancies[i] = get_occupancy_subset(i, num_mask_bits, blocker_mask);
        if (is_rook) {
            attacks[i] = calculate_rook_attacks_on_the_fly(sq, occupancies[i]);
        } else {
            attacks[i] = calculate_bishop_attacks_on_the_fly(sq, occupancies[i]);
        }
    }

    // 2. Find a magic number
    std::vector<Bitboard> temp_attack_table(num_occupancy_permutations, 0ULL);
    std::vector<bool>        temp_table_used(num_occupancy_permutations, false);

    for (int try_count = 0; try_count < 1000000; ++try_count) {
        Bitboard magic_candidate = generate_random_magic_candidate(); // Uses local finder's generator
        
        if (pop_count((blocker_mask * magic_candidate) & 0xFF00000000000000ULL) < 6) {
             continue;
        }

        std::fill(temp_table_used.begin(), temp_table_used.end(), false);
        bool possible_magic = true;

        for (size_t i = 0; i < num_occupancy_permutations; ++i) {
            unsigned int magic_index = (occupancies[i] * magic_candidate) >> shift_bits;

            if (!temp_table_used[magic_index]) {
                temp_attack_table[magic_index] = attacks[i];
                temp_table_used[magic_index] = true;
            } else if (temp_attack_table[magic_index] != attacks[i]) {
                possible_magic = false;
                break; 
            }
        }

        if (possible_magic) {
            return magic_candidate; 
        }
    }

    std::cerr << "!!! FAILED to find magic for sq " << sq << (is_rook ? " (ROOK)" : " (BISHOP)")
              << " after many tries." << std::endl;
    return 0ULL; // Indicate failure
}


int main() {
    std::cout << "Generating Magic Bitboard Numbers...\n" << std::endl;

    // Use chaturaji_cpp::Bitboard and chaturaji_cpp::magic_utils::NUM_SQUARES
    std::array<chaturaji_cpp::Bitboard, chaturaji_cpp::magic_utils::NUM_SQUARES> rook_magics;
    std::array<int, chaturaji_cpp::magic_utils::NUM_SQUARES> rook_shifts;
    std::array<chaturaji_cpp::Bitboard, chaturaji_cpp::magic_utils::NUM_SQUARES> bishop_magics;
    std::array<int, chaturaji_cpp::magic_utils::NUM_SQUARES> bishop_shifts;

    std::vector<chaturaji_cpp::Bitboard> occupancies_temp;
    std::vector<chaturaji_cpp::Bitboard> attacks_temp;

    std::cout << "// --- ROOK MAGICS ---" << std::endl;
    // Note: NUM_SQUARES_BB should be chaturaji_cpp::magic_utils::NUM_SQUARES for consistency in output
    std::cout << "const std::array<Bitboard, " << chaturaji_cpp::magic_utils::NUM_SQUARES << "> RookMagics = {" << std::endl;
    for (int sq = 0; sq < chaturaji_cpp::magic_utils::NUM_SQUARES; ++sq) {
        rook_magics[sq] = find_magic_for_square(sq, true, occupancies_temp, attacks_temp, rook_shifts[sq]);
        std::cout << "    0x" << std::hex << std::setw(16) << std::setfill('0') << rook_magics[sq] << "ULL,"
                  << " // sq " << std::dec << sq << " (shift " << rook_shifts[sq] << ")" << std::endl;
        if (rook_magics[sq] == 0ULL) {
             std::cerr << "CRITICAL: Could not find rook magic for square " << sq << std::endl;
             return 1;
        }
    }
    std::cout << "};" << std::endl << std::endl;


    std::cout << "// --- BISHOP MAGICS ---" << std::endl;
    std::cout << "const std::array<Bitboard, " << chaturaji_cpp::magic_utils::NUM_SQUARES << "> BishopMagics = {" << std::endl;
    for (int sq = 0; sq < chaturaji_cpp::magic_utils::NUM_SQUARES; ++sq) {
        bishop_magics[sq] = find_magic_for_square(sq, false, occupancies_temp, attacks_temp, bishop_shifts[sq]);
        std::cout << "    0x" << std::hex << std::setw(16) << std::setfill('0') << bishop_magics[sq] << "ULL,"
                  << " // sq " << std::dec << sq << " (shift " << bishop_shifts[sq] << ")" << std::endl;
         if (bishop_magics[sq] == 0ULL) {
             std::cerr << "CRITICAL: Could not find bishop magic for square " << sq << std::endl;
             return 1;
        }
    }
    std::cout << "};" << std::endl << std::endl;

    std::cout << "// --- ROOK SHIFTS (can be derived, but useful for verification) ---" << std::endl;
    std::cout << "const std::array<int, " << chaturaji_cpp::magic_utils::NUM_SQUARES << "> RookShifts = {" << std::endl;
    for(int sq=0; sq < chaturaji_cpp::magic_utils::NUM_SQUARES; ++sq) {
        std::cout << "    " << std::setw(2) << rook_shifts[sq] << ((sq % 8 == 7) ? ",\n" : ", "); // Added setw for alignment
    }
    std::cout << "};" << std::endl << std::endl;

    std::cout << "// --- BISHOP SHIFTS (can be derived, but useful for verification) ---" << std::endl;
    std::cout << "const std::array<int, " << chaturaji_cpp::magic_utils::NUM_SQUARES << "> BishopShifts = {" << std::endl;
    for(int sq=0; sq < chaturaji_cpp::magic_utils::NUM_SQUARES; ++sq) {
        std::cout << "    " << std::setw(2) << bishop_shifts[sq] << ((sq % 8 == 7) ? ",\n" : ", "); // Added setw for alignment
    }
    std::cout << "};" << std::endl << std::endl;


    std::cout << "\nMagic number generation complete." << std::endl;
    std::cout << "Remember to copy these generated arrays into magic_utils.cpp "
              << "and ensure magic_utils.h declares them extern." << std::endl;
    std::cout << "Your Board class will then use these values from magic_utils." << std::endl;

    return 0;
}