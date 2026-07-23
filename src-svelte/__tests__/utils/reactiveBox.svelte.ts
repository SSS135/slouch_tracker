// Minimal reactive box for tests: exposes a rune-backed value a component's $derived
// tracks, so a mocked query result can flip and drive the component's effects. Runes
// only compile in .svelte.ts, so query mocks that must change post-mount use this.
export function reactiveBox<T>(initial: T): { readonly value: T; set: (next: T) => void } {
  let value = $state(initial);
  return {
    get value() { return value; },
    set: (next: T) => { value = next; },
  };
}
