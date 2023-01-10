export const config = {
  // Number of heaps to pick the pins from
  heaps: 3,
  // Number of pins in each heap for evenly distributed games. Otherwise, the number of pins for the last heap
  heapSize: 3,
  // If heaps are evenly distributed each heap has the same number of pins at start.
  // Otherwise, the heaps are filled in an increasing order ending with heapSize pins (ex. 1, 2, 3, ..., heapSize).
  // If heaps and heapSize are different, the first heap will contain more than one pin
  evenDistributed: false,
  // For misère games the player picking the last pin loose. For normal games you win as you clear the board
  misereGame: true
}
