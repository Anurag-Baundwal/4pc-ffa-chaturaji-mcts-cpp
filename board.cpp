// board.cpp
#include "board.h"
#include "magic_utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chaturaji_cpp {

namespace { 
const int NUM_BB_PIECE_TYPES = 5;
const int NUM_PLAYERS_BB = 4;

struct ZobristData {
  std::array<std::array<std::array<ZobristKey, magic_utils::NUM_SQUARES>, NUM_PLAYERS_BB>, NUM_BB_PIECE_TYPES> piece_keys;
  std::array<ZobristKey, NUM_PLAYERS_BB> turn_keys;
  std::array<ZobristKey, NUM_PLAYERS_BB> active_player_status_keys;
  ZobristData() {
    std::mt19937_64 rng(0xBADFACE);
    std::uniform_int_distribution<ZobristKey> dist(0, std::numeric_limits<ZobristKey>::max());

    for (int type_idx = 0; type_idx < NUM_BB_PIECE_TYPES; ++type_idx) {
      for (int player_idx = 0; player_idx < NUM_PLAYERS_BB; ++player_idx) {
          for (int sq_idx = 0; sq_idx < magic_utils::NUM_SQUARES; ++sq_idx) {
            piece_keys[type_idx][player_idx][sq_idx] = dist(rng);
          }
      }
    }
    for (int player_idx = 0; player_idx < NUM_PLAYERS_BB; ++player_idx) {
      turn_keys[player_idx] = dist(rng);
      active_player_status_keys[player_idx] = dist(rng);
    }
  }

  ZobristKey get_piece_key(PieceType type, Player player, int square_index) const {
    int type_idx = static_cast<int>(type) - 1;
    int player_idx = static_cast<int>(player);
    return piece_keys[type_idx][player_idx][square_index]; 
  }

  ZobristKey get_turn_key(Player player) const {
    return turn_keys[static_cast<int>(player)];
  }

  ZobristKey get_active_player_status_key(Player player) const {
    return active_player_status_keys[static_cast<int>(player)];
  }
};

const ZobristData &get_zobrist_data() {
  static const ZobristData instance;
  return instance;
}

const int PROMOTION_ROW_RED_BB = 0;
const int PROMOTION_COL_BLUE_BB = 7;
const int PROMOTION_ROW_YELLOW_BB = 7;
const int PROMOTION_COL_GREEN_BB = 0;
}

std::array<Bitboard, magic_utils::NUM_SQUARES> Board::knight_attacks_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::king_attacks_;
std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> Board::pawn_attacks_red_;
std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> Board::pawn_attacks_blue_;
std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> Board::pawn_attacks_yellow_;
std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> Board::pawn_attacks_green_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::pawn_fwd_moves_red_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::pawn_fwd_moves_blue_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::pawn_fwd_moves_yellow_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::pawn_fwd_moves_green_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::rook_masks_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::bishop_masks_;
std::array<int, magic_utils::NUM_SQUARES> Board::rook_shift_bits_;
std::array<int, magic_utils::NUM_SQUARES> Board::bishop_shift_bits_;
std::vector<Bitboard> Board::rook_attack_table_;
std::vector<Bitboard> Board::bishop_attack_table_;
std::array<unsigned int, magic_utils::NUM_SQUARES> Board::rook_attack_offsets_;
std::array<unsigned int, magic_utils::NUM_SQUARES> Board::bishop_attack_offsets_;
Board::StaticInitializer Board::static_initializer_; 

int Board::piece_type_to_bb_idx(PieceType pt) {
    return static_cast<int>(pt) - 1;
}

bool Board::is_valid_sq_idx(int sq_idx) {
    return sq_idx >= 0 && sq_idx < magic_utils::NUM_SQUARES;
}

void Board::initialize_lookup_tables() {
    const int kn_moves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
    const int ki_moves[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int sq = magic_utils::to_sq_idx(r, c);
            knight_attacks_[sq] = 0; king_attacks_[sq] = 0;
            for (auto& m : kn_moves) {
                int nr = r + m[0], nc = c + m[1];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) magic_utils::set_bit(knight_attacks_[sq], magic_utils::to_sq_idx(nr, nc));
            }
            for (auto& m : ki_moves) {
                int nr = r + m[0], nc = c + m[1];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) magic_utils::set_bit(king_attacks_[sq], magic_utils::to_sq_idx(nr, nc));
            }
            pawn_fwd_moves_red_[sq] = 0; pawn_fwd_moves_blue_[sq] = 0; pawn_fwd_moves_yellow_[sq] = 0; pawn_fwd_moves_green_[sq] = 0;
            if (r > 0) {
                magic_utils::set_bit(pawn_fwd_moves_red_[sq], magic_utils::to_sq_idx(r-1, c));
                if (c > 0) magic_utils::set_bit(pawn_attacks_red_[0][sq], magic_utils::to_sq_idx(r-1, c-1));
                if (c < 7) magic_utils::set_bit(pawn_attacks_red_[0][sq], magic_utils::to_sq_idx(r-1, c+1));
            }
            if (c < 7) {
                magic_utils::set_bit(pawn_fwd_moves_blue_[sq], magic_utils::to_sq_idx(r, c+1));
                if (r > 0) magic_utils::set_bit(pawn_attacks_blue_[1][sq], magic_utils::to_sq_idx(r-1, c+1));
                if (r < 7) magic_utils::set_bit(pawn_attacks_blue_[1][sq], magic_utils::to_sq_idx(r+1, c+1));
            }
            if (r < 7) {
                magic_utils::set_bit(pawn_fwd_moves_yellow_[sq], magic_utils::to_sq_idx(r+1, c));
                if (c > 0) magic_utils::set_bit(pawn_attacks_yellow_[2][sq], magic_utils::to_sq_idx(r+1, c-1));
                if (c < 7) magic_utils::set_bit(pawn_attacks_yellow_[2][sq], magic_utils::to_sq_idx(r+1, c+1));
            }
            if (c > 0) {
                magic_utils::set_bit(pawn_fwd_moves_green_[sq], magic_utils::to_sq_idx(r, c-1));
                if (r > 0) magic_utils::set_bit(pawn_attacks_green_[3][sq], magic_utils::to_sq_idx(r-1, c-1));
                if (r < 7) magic_utils::set_bit(pawn_attacks_green_[3][sq], magic_utils::to_sq_idx(r+1, c-1));
            }
        }
    }
    unsigned int tr = 0, tb = 0;
    for (int sq = 0; sq < 64; ++sq) {
        rook_masks_[sq] = magic_utils::generate_rook_mask(sq);
        bishop_masks_[sq] = magic_utils::generate_bishop_mask(sq);
        rook_shift_bits_[sq] = magic_utils::RookShifts[sq];
        bishop_shift_bits_[sq] = magic_utils::BishopShifts[sq];
        rook_attack_offsets_[sq] = tr; tr += (1ULL << magic_utils::pop_count(rook_masks_[sq]));
        bishop_attack_offsets_[sq] = tb; tb += (1ULL << magic_utils::pop_count(bishop_masks_[sq]));
    }
    rook_attack_table_.resize(tr); bishop_attack_table_.resize(tb);
    for (int sq = 0; sq < 64; ++sq) {
        int r_bits = magic_utils::pop_count(rook_masks_[sq]);
        for (unsigned int i = 0; i < (1U << r_bits); ++i) {
            Bitboard occ = magic_utils::get_occupancy_subset(i, r_bits, rook_masks_[sq]);
            unsigned int idx = (occ * magic_utils::RookMagics[sq]) >> rook_shift_bits_[sq];
            rook_attack_table_[rook_attack_offsets_[sq] + idx] = magic_utils::calculate_rook_attacks_on_the_fly(sq, occ);
        }
        int b_bits = magic_utils::pop_count(bishop_masks_[sq]);
        for (unsigned int i = 0; i < (1U << b_bits); ++i) {
            Bitboard occ = magic_utils::get_occupancy_subset(i, b_bits, bishop_masks_[sq]);
            unsigned int idx = (occ * magic_utils::BishopMagics[sq]) >> bishop_shift_bits_[sq];
            bishop_attack_table_[bishop_attack_offsets_[sq] + idx] = magic_utils::calculate_bishop_attacks_on_the_fly(sq, occ);
        }
    }
}

// --- Constructors and Assignment Operators ---

Board::Board() : active_mask_(0xF), current_player_(Player::RED), full_move_number_(1), move_number_of_last_reset_(0), current_hash_(0) {
  player_points_.fill(0);
  for (auto& arr : piece_bitboards_) arr.fill(0ULL);
  player_bitboards_.fill(0ULL); combined_piece_bitboards_.fill(0ULL); occupied_bitboard_ = 0ULL;
  mailbox_.fill(std::nullopt);
  setup_initial_board();
  const auto& zd = get_zobrist_data();
  for (int p = 0; p < 4; ++p) {
      for (int pt = 0; pt < 5; ++pt) {
          Bitboard bb = piece_bitboards_[p][pt];
          while (bb) current_hash_ ^= zd.piece_keys[pt][p][magic_utils::pop_lsb(bb)];
      }
      current_hash_ ^= zd.get_active_player_status_key(static_cast<Player>(p));
  }
  current_hash_ ^= zd.get_turn_key(current_player_);
  position_history_.push_back(current_hash_);
}

Board::Board(const Board &o) : active_mask_(o.active_mask_), player_points_(o.player_points_), current_player_(o.current_player_), position_history_(o.position_history_), full_move_number_(o.full_move_number_), move_number_of_last_reset_(o.move_number_of_last_reset_), termination_reason_(o.termination_reason_), current_hash_(o.current_hash_), undo_stack_(o.undo_stack_), piece_bitboards_(o.piece_bitboards_), player_bitboards_(o.player_bitboards_), occupied_bitboard_(o.occupied_bitboard_), combined_piece_bitboards_(o.combined_piece_bitboards_), mailbox_(o.mailbox_) {}
Board::Board(const Board &o, MCTSChildCopyTag) : active_mask_(o.active_mask_), player_points_(o.player_points_), current_player_(o.current_player_), position_history_(o.position_history_), full_move_number_(o.full_move_number_), move_number_of_last_reset_(o.move_number_of_last_reset_), termination_reason_(std::nullopt), current_hash_(o.current_hash_), piece_bitboards_(o.piece_bitboards_), player_bitboards_(o.player_bitboards_), occupied_bitboard_(o.occupied_bitboard_), combined_piece_bitboards_(o.combined_piece_bitboards_), mailbox_(o.mailbox_) {}
Board::Board(Board &&o) noexcept : active_mask_(o.active_mask_), player_points_(std::move(o.player_points_)), current_player_(o.current_player_), position_history_(std::move(o.position_history_)), full_move_number_(o.full_move_number_), move_number_of_last_reset_(o.move_number_of_last_reset_), termination_reason_(std::move(o.termination_reason_)), current_hash_(o.current_hash_), undo_stack_(std::move(o.undo_stack_)), piece_bitboards_(std::move(o.piece_bitboards_)), player_bitboards_(std::move(o.player_bitboards_)), occupied_bitboard_(o.occupied_bitboard_), combined_piece_bitboards_(std::move(o.combined_piece_bitboards_)), mailbox_(std::move(o.mailbox_)) {}

Board &Board::operator=(const Board &o) {
  if (this != &o) { active_mask_ = o.active_mask_; player_points_ = o.player_points_; current_player_ = o.current_player_; position_history_ = o.position_history_; full_move_number_ = o.full_move_number_; move_number_of_last_reset_ = o.move_number_of_last_reset_; termination_reason_ = o.termination_reason_; current_hash_ = o.current_hash_; undo_stack_ = o.undo_stack_; piece_bitboards_ = o.piece_bitboards_; player_bitboards_ = o.player_bitboards_; occupied_bitboard_ = o.occupied_bitboard_; combined_piece_bitboards_ = o.combined_piece_bitboards_; mailbox_ = o.mailbox_; }
  return *this;
}
Board &Board::operator=(Board &&o) noexcept {
  if (this != &o) { active_mask_ = o.active_mask_; player_points_ = std::move(o.player_points_); current_player_ = o.current_player_; position_history_ = std::move(o.position_history_); full_move_number_ = o.full_move_number_; move_number_of_last_reset_ = o.move_number_of_last_reset_; termination_reason_ = std::move(o.termination_reason_); current_hash_ = o.current_hash_; undo_stack_ = std::move(o.undo_stack_); piece_bitboards_ = std::move(o.piece_bitboards_); player_bitboards_ = std::move(o.player_bitboards_); occupied_bitboard_ = o.occupied_bitboard_; combined_piece_bitboards_ = std::move(o.combined_piece_bitboards_); mailbox_ = std::move(o.mailbox_); }
  return *this;
}

Board Board::create_mcts_child_board(const Board& p, const Move& m) { Board c(p, MCTSChildCopyTag{}); c.make_move_for_mcts(m); return c; }

const std::set<Player> Board::get_active_players() const {
    std::set<Player> res; for (int i = 0; i < 4; ++i) if (active_mask_ & (1 << i)) res.insert(static_cast<Player>(i)); return res;
}

// Optimized O(1) piece lookup
std::optional<Piece> Board::get_piece_at_sq(int sq) const {
    if (!is_valid_sq_idx(sq)) return std::nullopt;
    return mailbox_[sq];
}

void Board::setup_initial_board() {
  auto place = [&](Player p, PieceType pt, int r, int c) {
      int sq = magic_utils::to_sq_idx(r, c), pi = static_cast<int>(p), ti = piece_type_to_bb_idx(pt);
      magic_utils::set_bit(piece_bitboards_[pi][ti], sq); magic_utils::set_bit(player_bitboards_[pi], sq);
      magic_utils::set_bit(combined_piece_bitboards_[ti], sq); magic_utils::set_bit(occupied_bitboard_, sq);
      mailbox_[sq] = Piece(p, pt);
  };
  place(Player::RED, PieceType::ROOK, 7, 0); place(Player::RED, PieceType::KNIGHT, 7, 1); place(Player::RED, PieceType::BISHOP, 7, 2); place(Player::RED, PieceType::KING, 7, 3);
  for (int i = 0; i < 4; ++i) place(Player::RED, PieceType::PAWN, 6, i);
  place(Player::BLUE, PieceType::ROOK, 0, 0); place(Player::BLUE, PieceType::KNIGHT, 1, 0); place(Player::BLUE, PieceType::BISHOP, 2, 0); place(Player::BLUE, PieceType::KING, 3, 0);
  for (int i = 0; i < 4; ++i) place(Player::BLUE, PieceType::PAWN, i, 1);
  place(Player::YELLOW, PieceType::ROOK, 0, 7); place(Player::YELLOW, PieceType::KNIGHT, 0, 6); place(Player::YELLOW, PieceType::BISHOP, 0, 5); place(Player::YELLOW, PieceType::KING, 0, 4);
  for (int i = 4; i < 8; ++i) place(Player::YELLOW, PieceType::PAWN, 1, i);
  place(Player::GREEN, PieceType::KING, 4, 7); place(Player::GREEN, PieceType::BISHOP, 5, 7); place(Player::GREEN, PieceType::KNIGHT, 6, 7); place(Player::GREEN, PieceType::ROOK, 7, 7);
  for (int i = 4; i < 8; ++i) place(Player::GREEN, PieceType::PAWN, i, 6);
}

bool Board::is_valid_square(int r, int c) const { return r >= 0 && r < 8 && c >= 0 && c < 8; }

void Board::get_pseudo_legal_moves(Player p, MoveList& moves) const {
  moves.clear();
  if (!(active_mask_ & (1 << static_cast<int>(p)))) return;
  moves.push_back(Move::Resign());
  get_pawn_moves_bb(p, moves); get_knight_moves_bb(p, moves); get_bishop_moves_bb(p, moves); get_rook_moves_bb(p, moves); get_king_moves_bb(p, moves);
}

// Legacy wrapper
std::vector<Move> Board::get_pseudo_legal_moves(Player p) const {
    MoveList moves;
    get_pseudo_legal_moves(p, moves);
    std::vector<Move> res;
    res.reserve(moves.size());
    for(size_t i=0; i<moves.size(); ++i) res.push_back(moves[i]);
    return res;
}

void Board::get_pawn_moves_bb(Player p, MoveList& m) const {
    int pi = static_cast<int>(p); Bitboard pawns = piece_bitboards_[pi][0], empty = ~occupied_bitboard_, opp = occupied_bitboard_ & ~player_bitboards_[pi];
    const Bitboard* fwd = nullptr; const std::array<Bitboard, 64>* atk = nullptr; int trg = -1; bool is_r = false;
    switch (p) {
        case Player::RED: fwd = &pawn_fwd_moves_red_[0]; atk = &pawn_attacks_red_[0]; trg = 0; is_r = true; break;
        case Player::BLUE: fwd = &pawn_fwd_moves_blue_[0]; atk = &pawn_attacks_blue_[1]; trg = 7; is_r = false; break;
        case Player::YELLOW: fwd = &pawn_fwd_moves_yellow_[0]; atk = &pawn_attacks_yellow_[2]; trg = 7; is_r = true; break;
        case Player::GREEN: fwd = &pawn_fwd_moves_green_[0]; atk = &pawn_attacks_green_[3]; trg = 0; is_r = false; break;
    }
    while (pawns) {
        int fs = magic_utils::pop_lsb(pawns); BoardLocation fl = magic_utils::from_sq_idx(fs);
        Bitboard f = fwd[fs] & empty; if (f) {
            int ts = magic_utils::get_lsb_index(f); BoardLocation tl = magic_utils::from_sq_idx(ts);
            if ((is_r && tl.row == trg) || (!is_r && tl.col == trg)) m.push_back(Move(fl, tl, PieceType::ROOK)); else m.push_back(Move(fl, tl));
        }
        Bitboard c = (*atk)[fs] & opp;
        while (c) {
            int ts = magic_utils::pop_lsb(c); BoardLocation tl = magic_utils::from_sq_idx(ts);
            if ((is_r && tl.row == trg) || (!is_r && tl.col == trg)) m.push_back(Move(fl, tl, PieceType::ROOK)); else m.push_back(Move(fl, tl));
        }
    }
}

void Board::get_knight_moves_bb(Player p, MoveList& m) const {
    int pi = static_cast<int>(p); Bitboard knights = piece_bitboards_[pi][1], targets = ~player_bitboards_[pi];
    while (knights) {
        int fs = magic_utils::pop_lsb(knights); BoardLocation fl = magic_utils::from_sq_idx(fs); Bitboard b = knight_attacks_[fs] & targets;
        while (b) m.push_back(Move(fl, magic_utils::from_sq_idx(magic_utils::pop_lsb(b))));
    }
}

void Board::get_king_moves_bb(Player p, MoveList& m) const {
    int pi = static_cast<int>(p); Bitboard kings = piece_bitboards_[pi][4], targets = ~player_bitboards_[pi];
    if (!kings) return;
    int fs = magic_utils::get_lsb_index(kings); BoardLocation fl = magic_utils::from_sq_idx(fs); Bitboard b = king_attacks_[fs] & targets;
    while (b) m.push_back(Move(fl, magic_utils::from_sq_idx(magic_utils::pop_lsb(b))));
}

void Board::get_rook_moves_bb(Player p, MoveList& m) const {
    int pi = static_cast<int>(p); Bitboard rooks = piece_bitboards_[pi][3], my = player_bitboards_[pi];
    while(rooks) {
        int fs = magic_utils::pop_lsb(rooks); BoardLocation fl = magic_utils::from_sq_idx(fs);
        Bitboard b = occupied_bitboard_ & rook_masks_[fs];
        Bitboard pos = rook_attack_table_[rook_attack_offsets_[fs] + ((b * magic_utils::RookMagics[fs]) >> rook_shift_bits_[fs])] & ~my;
        while(pos) m.push_back(Move(fl, magic_utils::from_sq_idx(magic_utils::pop_lsb(pos))));
    }
}

void Board::get_bishop_moves_bb(Player p, MoveList& m) const {
    int pi = static_cast<int>(p); Bitboard bishops = piece_bitboards_[pi][2], my = player_bitboards_[pi];
    while(bishops) {
        int fs = magic_utils::pop_lsb(bishops); BoardLocation fl = magic_utils::from_sq_idx(fs);
        Bitboard b = occupied_bitboard_ & bishop_masks_[fs];
        Bitboard pos = bishop_attack_table_[bishop_attack_offsets_[fs] + ((b * magic_utils::BishopMagics[fs]) >> bishop_shift_bits_[fs])] & ~my;
        while(pos) m.push_back(Move(fl, magic_utils::from_sq_idx(magic_utils::pop_lsb(pos))));
    }
}

std::optional<Piece> Board::make_move(const Move &m) {
  UndoInfo ui; ui.original_piece_bitboards = piece_bitboards_; ui.original_player_bitboards = player_bitboards_; ui.original_occupied_bitboard = occupied_bitboard_; ui.original_combined_piece_bitboards = combined_piece_bitboards_; ui.original_active_mask = active_mask_; ui.original_mailbox = mailbox_;
  ui.move = m; ui.original_player = current_player_; ui.original_full_move_number = full_move_number_; ui.original_move_number_of_last_reset = move_number_of_last_reset_; ui.eliminated_player = std::nullopt; ui.was_history_cleared = false; ui.previous_hash = current_hash_; ui.check_bonus_points = 0;
  
  if (m.is_resignation()) {
      if (active_mask_ & (1 << static_cast<int>(current_player_))) {
          eliminate_player(current_player_);
          ui.eliminated_player = current_player_;
      }
      advance_turn();
      undo_stack_.push_back(std::move(ui));
      return std::nullopt;
  }

  struct EKS { Player p; int sq; Bitboard atks; }; std::vector<EKS> eks; eks.reserve(3);
  for (int i=0; i<4; ++i) if ((active_mask_ & (1<<i)) && static_cast<Player>(i) != current_player_) {
      Bitboard k = piece_bitboards_[i][4]; if (k) eks.push_back({static_cast<Player>(i), magic_utils::get_lsb_index(k), get_attackers_on_sq(magic_utils::get_lsb_index(k))});
  }
  const auto& zd = get_zobrist_data(); int fs = magic_utils::to_sq_idx(m.from_loc.row, m.from_loc.col), ts = magic_utils::to_sq_idx(m.to_loc.row, m.to_loc.col), pi = static_cast<int>(current_player_);
  std::optional<Piece> mp = get_piece_at_sq(fs); if (!mp) throw std::runtime_error("Empty square");
  int ti = piece_type_to_bb_idx(mp->piece_type);
  std::optional<Piece> cap = get_piece_at_sq(ts); bool is_cap = cap.has_value();
  ui.captured_piece = cap;
  current_hash_ ^= zd.get_piece_key(mp->piece_type, mp->player, fs);
  magic_utils::clear_bit(piece_bitboards_[pi][ti], fs); magic_utils::clear_bit(combined_piece_bitboards_[ti], fs); magic_utils::clear_bit(player_bitboards_[pi], fs); magic_utils::clear_bit(occupied_bitboard_, fs);
  mailbox_[fs] = std::nullopt;

  if (is_cap) {
      current_hash_ ^= zd.get_piece_key(cap->piece_type, cap->player, ts);
      int ci = static_cast<int>(cap->player), cti = piece_type_to_bb_idx(cap->piece_type);
      magic_utils::clear_bit(piece_bitboards_[ci][cti], ts); magic_utils::clear_bit(combined_piece_bitboards_[cti], ts); magic_utils::clear_bit(player_bitboards_[ci], ts);
      player_points_[pi] += get_piece_capture_value(*cap); if (cap->piece_type == PieceType::KING) { eliminate_player(cap->player); ui.eliminated_player = cap->player; }
  }
  PieceType ft = m.promotion_piece_type ? *m.promotion_piece_type : mp->piece_type;
  int fti = piece_type_to_bb_idx(ft);
  magic_utils::set_bit(piece_bitboards_[pi][fti], ts); magic_utils::set_bit(combined_piece_bitboards_[fti], ts); magic_utils::set_bit(player_bitboards_[pi], ts); magic_utils::set_bit(occupied_bitboard_, ts);
  mailbox_[ts] = Piece(mp->player, ft);
  current_hash_ ^= zd.get_piece_key(ft, mp->player, ts);
  int checked = 0; for (const auto& k : eks) if (active_mask_ & (1 << static_cast<int>(k.p))) {
      Bitboard kb = piece_bitboards_[static_cast<int>(k.p)][4]; if (kb && (get_attackers_on_sq(magic_utils::get_lsb_index(kb)) & ~k.atks)) checked++;
  }
  if (checked == 2) { player_points_[pi] += 1; ui.check_bonus_points = 1; } else if (checked == 3) { player_points_[pi] += 5; ui.check_bonus_points = 5; }
  if (is_cap || mp->piece_type == PieceType::PAWN) { position_history_.clear(); ui.was_history_cleared = true; move_number_of_last_reset_ = full_move_number_; }
  current_hash_ ^= zd.get_turn_key(current_player_); advance_turn(); current_hash_ ^= zd.get_turn_key(current_player_);
  position_history_.push_back(current_hash_); undo_stack_.push_back(std::move(ui)); return cap;
}

std::optional<Piece> Board::make_move_for_mcts(const Move &m) {
  if (m.is_resignation()) {
      if (active_mask_ & (1 << static_cast<int>(current_player_))) {
          eliminate_player(current_player_);
      }
      advance_turn();
      return std::nullopt;
  }
  struct EKS { Player p; int sq; Bitboard atks; }; std::vector<EKS> eks; eks.reserve(3);
  for (int i=0; i<4; ++i) if ((active_mask_ & (1<<i)) && static_cast<Player>(i) != current_player_) {
      Bitboard k = piece_bitboards_[i][4]; if (k) eks.push_back({static_cast<Player>(i), magic_utils::get_lsb_index(k), get_attackers_on_sq(magic_utils::get_lsb_index(k))});
  }
  const auto& zd = get_zobrist_data(); int fs = magic_utils::to_sq_idx(m.from_loc.row, m.from_loc.col), ts = magic_utils::to_sq_idx(m.to_loc.row, m.to_loc.col), pi = static_cast<int>(current_player_);
  std::optional<Piece> mp = get_piece_at_sq(fs); if (!mp) throw std::runtime_error("Empty square");
  int ti = piece_type_to_bb_idx(mp->piece_type);
  std::optional<Piece> cap = get_piece_at_sq(ts); bool is_cap = cap.has_value();
  current_hash_ ^= zd.get_piece_key(mp->piece_type, mp->player, fs);
  magic_utils::clear_bit(piece_bitboards_[pi][ti], fs); magic_utils::clear_bit(combined_piece_bitboards_[ti], fs); magic_utils::clear_bit(player_bitboards_[pi], fs); magic_utils::clear_bit(occupied_bitboard_, fs);
  mailbox_[fs] = std::nullopt;

  if (is_cap) {
      current_hash_ ^= zd.get_piece_key(cap->piece_type, cap->player, ts);
      int ci = static_cast<int>(cap->player), cti = piece_type_to_bb_idx(cap->piece_type);
      magic_utils::clear_bit(piece_bitboards_[ci][cti], ts); magic_utils::clear_bit(combined_piece_bitboards_[cti], ts); magic_utils::clear_bit(player_bitboards_[ci], ts);
      player_points_[pi] += get_piece_capture_value(*cap); if (cap->piece_type == PieceType::KING) eliminate_player(cap->player);
  }
  PieceType ft = m.promotion_piece_type ? *m.promotion_piece_type : mp->piece_type;
  int fti = piece_type_to_bb_idx(ft);
  magic_utils::set_bit(piece_bitboards_[pi][fti], ts); magic_utils::set_bit(combined_piece_bitboards_[fti], ts); magic_utils::set_bit(player_bitboards_[pi], ts); magic_utils::set_bit(occupied_bitboard_, ts);
  mailbox_[ts] = Piece(mp->player, ft);
  current_hash_ ^= zd.get_piece_key(ft, mp->player, ts);
  int checked = 0; for (const auto& k : eks) if (active_mask_ & (1 << static_cast<int>(k.p))) {
      Bitboard kb = piece_bitboards_[static_cast<int>(k.p)][4]; if (kb && (get_attackers_on_sq(magic_utils::get_lsb_index(kb)) & ~k.atks)) checked++;
  }
  if (checked == 2) player_points_[pi] += 1; else if (checked == 3) player_points_[pi] += 5;
  if (is_cap || mp->piece_type == PieceType::PAWN) { position_history_.clear(); move_number_of_last_reset_ = full_move_number_; }
  current_hash_ ^= zd.get_turn_key(current_player_); advance_turn(); current_hash_ ^= zd.get_turn_key(current_player_);
  position_history_.push_back(current_hash_); return cap;
}

void Board::undo_move() {
  if (undo_stack_.empty()) throw std::runtime_error("No undo");
  UndoInfo ui = undo_stack_.back(); undo_stack_.pop_back();
  piece_bitboards_ = ui.original_piece_bitboards; player_bitboards_ = ui.original_player_bitboards; occupied_bitboard_ = ui.original_occupied_bitboard; combined_piece_bitboards_ = ui.original_combined_piece_bitboards; active_mask_ = ui.original_active_mask; mailbox_ = ui.original_mailbox;
  current_hash_ = ui.previous_hash; current_player_ = ui.original_player; full_move_number_ = ui.original_full_move_number; move_number_of_last_reset_ = ui.original_move_number_of_last_reset;
  if (!position_history_.empty()) position_history_.pop_back();
  if (!ui.move.is_resignation() && ui.captured_piece) player_points_[static_cast<int>(ui.original_player)] -= get_piece_capture_value(*ui.captured_piece);
  player_points_[static_cast<int>(ui.original_player)] -= ui.check_bonus_points;
  termination_reason_ = std::nullopt;
}

void Board::eliminate_player(Player p) {
  int pi = static_cast<int>(p); if (active_mask_ & (1 << pi)) { current_hash_ ^= get_zobrist_data().get_active_player_status_key(p); active_mask_ &= ~(1 << pi); }
}

Bitboard Board::get_occupied_bitboard() const { return occupied_bitboard_; }
Bitboard Board::get_player_bitboard(Player p) const { return player_bitboards_[static_cast<int>(p)]; }
Bitboard Board::get_piece_bitboard(Player p, PieceType pt) const { return piece_bitboards_[static_cast<int>(p)][piece_type_to_bb_idx(pt)]; }

Bitboard Board::get_squares_attacked_by(Player p) const {
  if (!(active_mask_ & (1 << static_cast<int>(p)))) return 0ULL;
  Bitboard a = 0ULL, occ = occupied_bitboard_; int pi = static_cast<int>(p);
  Bitboard pawns = piece_bitboards_[pi][0];
  while (pawns) {
      int sq = magic_utils::pop_lsb(pawns);
      if (p == Player::RED) a |= pawn_attacks_red_[0][sq]; else if (p == Player::BLUE) a |= pawn_attacks_blue_[1][sq]; else if (p == Player::YELLOW) a |= pawn_attacks_yellow_[2][sq]; else a |= pawn_attacks_green_[3][sq];
  }
  Bitboard knights = piece_bitboards_[pi][1]; while (knights) a |= knight_attacks_[magic_utils::pop_lsb(knights)];
  Bitboard king = piece_bitboards_[pi][4]; if (king) a |= king_attacks_[magic_utils::get_lsb_index(king)];
  Bitboard rooks = piece_bitboards_[pi][3]; while (rooks) {
      int sq = magic_utils::pop_lsb(rooks); Bitboard b = occ & rook_masks_[sq];
      a |= rook_attack_table_[rook_attack_offsets_[sq] + ((b * magic_utils::RookMagics[sq]) >> rook_shift_bits_[sq])];
  }
  Bitboard bishops = piece_bitboards_[pi][2]; while (bishops) {
      int sq = magic_utils::pop_lsb(bishops); Bitboard b = occ & bishop_masks_[sq];
      a |= bishop_attack_table_[bishop_attack_offsets_[sq] + ((b * magic_utils::BishopMagics[sq]) >> bishop_shift_bits_[sq])];
  }
  return a;
}

Bitboard Board::get_attackers_on_sq(int sq) const {
    Bitboard a = (knight_attacks_[sq] & combined_piece_bitboards_[1]) | (king_attacks_[sq] & combined_piece_bitboards_[4]);
    Bitboard rb = occupied_bitboard_ & rook_masks_[sq]; a |= (rook_attack_table_[rook_attack_offsets_[sq] + ((rb * magic_utils::RookMagics[sq]) >> rook_shift_bits_[sq])] & combined_piece_bitboards_[3]);
    Bitboard bb = occupied_bitboard_ & bishop_masks_[sq]; a |= (bishop_attack_table_[bishop_attack_offsets_[sq] + ((bb * magic_utils::BishopMagics[sq]) >> bishop_shift_bits_[sq])] & combined_piece_bitboards_[2]);
    if (active_mask_ & 1) a |= (pawn_attacks_yellow_[2][sq] & piece_bitboards_[0][0]);
    if (active_mask_ & 2) a |= (pawn_attacks_green_[3][sq] & piece_bitboards_[1][0]);
    if (active_mask_ & 4) a |= (pawn_attacks_red_[0][sq] & piece_bitboards_[2][0]);
    if (active_mask_ & 8) a |= (pawn_attacks_blue_[1][sq] & piece_bitboards_[3][0]);
    return a;
}

void Board::print_bitboard(Bitboard bb, const std::string& l) {
    std::cout << l << std::endl; for (int r = 0; r < 8; ++r) { for (int c = 0; c < 8; ++c) std::cout << (magic_utils::get_bit(bb, magic_utils::to_sq_idx(r, c)) ? "1 " : ". "); std::cout << std::endl; }
}

const Board::PlayerPointMap& Board::get_player_points() const { return player_points_; }
Player Board::get_current_player() const { return current_player_; }
int Board::get_full_move_number() const { return full_move_number_; }
int Board::get_move_number_of_last_reset() const { return move_number_of_last_reset_; }
const std::optional<std::string> &Board::get_termination_reason() const { return termination_reason_; }
const Board::PositionHistory &Board::get_position_history() const { return position_history_; }

Player Board::get_last_active_player() const {
    for (int i = 3; i >= 0; --i) if (active_mask_ & (1 << i)) return static_cast<Player>(i);
    return Player::GREEN;
}

bool Board::is_game_over() const {
  if (termination_reason_) return true;
  int n = magic_utils::pop_count(static_cast<Bitboard>(active_mask_));
  if (n == 2) {
    Player pc = current_player_, po = Player::RED;
    for (int i=0; i<4; ++i) if ((active_mask_ & (1<<i)) && static_cast<Player>(i) != pc) { po = static_cast<Player>(i); break; }
    int dk = 0; for (int i = 0; i < 4; ++i) if (!(active_mask_ & (1 << i)) && piece_bitboards_[i][4]) dk++;
    int pt = 3 + (3 * dk), sc = player_points_[static_cast<int>(pc)], so = player_points_[static_cast<int>(po)] + pt;
    bool abs = (sc > so); if (abs) { for (int i=0; i<4; ++i) if (static_cast<Player>(i) != pc && static_cast<Player>(i) != po && player_points_[i] >= sc) { abs = false; break; } }
    if (abs) {
      int bi = -1; for (int i=0; i<4; ++i) if (!(active_mask_ & (1 << i)) && player_points_[i] > bi) bi = player_points_[i];
      if (bi > player_points_[static_cast<int>(po)] && so <= bi) { termination_reason_ = "autoclaim"; return true; }
    }
  }
  if (n <= 1) { termination_reason_ = "elimination"; return true; }
  if (full_move_number_ - move_number_of_last_reset_ >= 50 && !undo_stack_.empty() && undo_stack_.back().original_player == get_last_active_player()) { termination_reason_ = "fifty_move_rule"; return true; }
  int c = 0; for (const auto &k : position_history_) if (k == current_hash_) c++;
  if (c >= 3) { termination_reason_ = "threefold_repetition"; return true; }
  return false;
}

std::map<Player, int> Board::get_game_result() const {
  std::map<Player, int> r; for(int i=0; i<4; ++i) r[static_cast<Player>(i)] = player_points_[i];
  int dk = 0; for (int i = 0; i < 4; ++i) if (!(active_mask_ & (1 << i)) && piece_bitboards_[i][4]) dk++;
  if (termination_reason_ && (*termination_reason_ == "fifty_move_rule" || *termination_reason_ == "threefold_repetition")) {
      for (int i=0; i<4; ++i) if (active_mask_ & (1 << i)) r[static_cast<Player>(i)] += (3 * dk);
  }
  return r;
}
std::optional<Player> Board::get_winner() const {
  if (!termination_reason_) return std::nullopt;
  auto s = get_game_result(); auto w = std::max_element(s.begin(), s.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
  return (w == s.end()) ? std::nullopt : std::optional<Player>(w->first);
}

int Board::get_piece_value(const Piece& p) const {
  switch (p.piece_type) { case PieceType::PAWN: return 1; case PieceType::KNIGHT: return 3; case PieceType::BISHOP: return 5; case PieceType::ROOK: return 5; case PieceType::KING: return 3; default: return 0; }
}
int Board::get_piece_capture_value(const Piece& p) const {
    if (!(active_mask_ & (1 << static_cast<int>(p.player)))) return (p.piece_type == PieceType::KING) ? 3 : 0;
    return get_piece_value(p);
}

std::map<Player, int> Board::evaluate() const {
  std::map<Player, int> e; for(int i=0; i<4; ++i) e[static_cast<Player>(i)] = player_points_[i];
  for (int p = 0; p < 4; ++p) {
    if (!(active_mask_ & (1 << p))) continue;
    for (int pt = 0; pt < 5; ++pt) e[static_cast<Player>(p)] += (magic_utils::pop_count(piece_bitboards_[p][pt]) * get_piece_value(Piece(static_cast<Player>(p), static_cast<PieceType>(pt + 1))));
  }
  return e;
}

void Board::resign() { if (is_player_active(current_player_)) eliminate_player(current_player_); advance_turn(); }

void Board::advance_turn() {
    if (active_mask_ == 0) return;
    int n = (static_cast<int>(current_player_) + 1) % 4;
    while (!(active_mask_ & (1 << n))) n = (n + 1) % 4;
    current_player_ = static_cast<Player>(n); if (current_player_ == Player::RED) full_move_number_++;
}

void Board::print_board() const {
  const std::string r = "\033[0m", cr = "\033[31m", cg = "\033[32m", cy = "\033[33m", cb = "\033[34m";
  const std::string sk = "♔", sr = "♖", sb = "♗", sn = "♘", sp = "♙";
  std::cout << "   a  b  c  d  e  f  g  h" << std::endl;
  for (int row = 0; row < 8; ++row) {
    std::cout << 8 - row << " ";
    for (int col = 0; col < 8; ++col) {
      auto p = get_piece_at_sq(magic_utils::to_sq_idx(row, col));
      std::string s = " ";
      if (p) {
        std::string bs; switch (p->piece_type) { case PieceType::PAWN: bs = sp; break; case PieceType::KNIGHT: bs = sn; break; case PieceType::BISHOP: bs = sb; break; case PieceType::ROOK: bs = sr; break; case PieceType::KING: bs = sk; break; }
        if (!(active_mask_ & (1 << static_cast<int>(p->player)))) s = bs;
        else { std::string c; switch (p->player) { case Player::RED: c = cr; break; case Player::BLUE: c = cb; break; case Player::YELLOW: c = cy; break; case Player::GREEN: c = cg; break; } s = c + bs + r; }
      }
      std::cout << "[" << s << "]";
    }
    std::cout << std::endl; 
  }
}

Board::PositionKey Board::get_position_key() const { return current_hash_; }

} // namespace chaturaji_cpp