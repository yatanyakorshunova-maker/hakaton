#pragma once

namespace generated_biomes {

inline void append(std::vector<const Biome*>& out) {
  static const GravityShearEscarpment biome_0; out.push_back(&biome_0);
}

inline constexpr int kGeneratedBiomeBankEnd = 0;

}
