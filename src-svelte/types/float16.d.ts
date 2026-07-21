interface Float16Array extends ArrayLike<number> {
  readonly BYTES_PER_ELEMENT: number;
  readonly buffer: ArrayBufferLike;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly length: number;
  [index: number]: number;
  copyWithin(target: number, start: number, end?: number): this;
  entries(): IterableIterator<[number, number]>;
  every(predicate: (value: number, index: number, array: Float16Array) => unknown): boolean;
  fill(value: number, start?: number, end?: number): this;
  filter(predicate: (value: number, index: number, array: Float16Array) => unknown): Float16Array;
  find(predicate: (value: number, index: number, obj: Float16Array) => boolean): number | undefined;
  findIndex(predicate: (value: number, index: number, obj: Float16Array) => boolean): number;
  forEach(callbackfn: (value: number, index: number, array: Float16Array) => void): void;
  includes(searchElement: number, fromIndex?: number): boolean;
  indexOf(searchElement: number, fromIndex?: number): number;
  join(separator?: string): string;
  keys(): IterableIterator<number>;
  lastIndexOf(searchElement: number, fromIndex?: number): number;
  map(callbackfn: (value: number, index: number, array: Float16Array) => number): Float16Array;
  reduce(callbackfn: (prev: number, curr: number, index: number, array: Float16Array) => number): number;
  reduce<U>(callbackfn: (prev: U, curr: number, index: number, array: Float16Array) => U, initialValue: U): U;
  reduceRight(callbackfn: (prev: number, curr: number, index: number, array: Float16Array) => number): number;
  reduceRight<U>(callbackfn: (prev: U, curr: number, index: number, array: Float16Array) => U, initialValue: U): U;
  reverse(): Float16Array;
  set(array: ArrayLike<number>, offset?: number): void;
  slice(start?: number, end?: number): Float16Array;
  some(predicate: (value: number, index: number, array: Float16Array) => unknown): boolean;
  sort(compareFn?: (a: number, b: number) => number): this;
  subarray(begin?: number, end?: number): Float16Array;
  toLocaleString(): string;
  toString(): string;
  valueOf(): Float16Array;
  values(): IterableIterator<number>;
  [Symbol.iterator](): IterableIterator<number>;
}

interface Float16ArrayConstructor {
  readonly prototype: Float16Array;
  new (length: number): Float16Array;
  new (array: ArrayLike<number> | ArrayBufferLike): Float16Array;
  new (buffer: ArrayBufferLike, byteOffset?: number, length?: number): Float16Array;
  readonly BYTES_PER_ELEMENT: number;
  of(...items: number[]): Float16Array;
  from(arrayLike: ArrayLike<number>): Float16Array;
  from<T>(arrayLike: ArrayLike<T>, mapfn: (v: T, k: number) => number, thisArg?: unknown): Float16Array;
}

declare var Float16Array: Float16ArrayConstructor;
