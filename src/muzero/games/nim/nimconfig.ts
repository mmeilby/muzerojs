export const config = {
  // Number of heaps to pick the pins from
  heaps: 5,
  // Number of pins in each heap for evenly distributed games. Otherwise, the number of pins for the last heap
  heapSize: 5,
  // If heaps are evenly distributed each heap has the same number of pins at start.
  // Otherwise, the heaps are filled in an increasing order ending with heapSize pins (ex. 1, 2, 3, ..., heapSize).
  // If heaps < heapSize, the first heap will contain more than one pin
  evenDistributed: false,
  // For misère games the player picking the last pin loose. For normal games you win as you clear the board
  misereGame: true
}

// GAME LOGIC - DO NOT CHANGE
export const util = {
  // Initial number of pins in each heap - ordered the smallest to the largest pin amount
  heapMap: new Array<number>(config.heaps).fill(config.heapSize).map((value, index) => config.evenDistributed ? value : Math.max(value - config.heaps + index + 1, 0))
}
