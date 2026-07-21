import { render, screen } from '@testing-library/svelte';
import Slider from '../Slider.svelte';

describe('Slider Component', () => {
  describe('Basic Rendering', () => {
    it('should display help text when provided', () => {
      render(Slider, {
        props: {
          label: 'Test Slider',
          value: 50,
          minimumValue: 0,
          maximumValue: 100,
          step: 1,
          onValueChange: () => {},
          helpText: 'This is helpful text',
        },
      });

      expect(screen.getByText('This is helpful text')).toBeInTheDocument();
    });
  });

  describe('Continuous Slider (no fixedValues)', () => {
    it('should work without fixedValues', () => {
      render(Slider, {
        props: {
          label: 'Continuous Slider',
          value: 50,
          minimumValue: 0,
          maximumValue: 100,
          step: 1,
          onValueChange: () => {},
        },
      });

      expect(screen.getByText('Continuous Slider')).toBeInTheDocument();
      expect(screen.getByDisplayValue('50')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty fixedValues array', () => {
      render(Slider, {
        props: {
          label: 'Test Slider',
          value: 50,
          minimumValue: 0,
          maximumValue: 100,
          step: 1,
          fixedValues: [],
          onValueChange: () => {},
        },
      });

      expect(screen.getByText('Test Slider')).toBeInTheDocument();
    });

    it('should handle single fixed value', () => {
      render(Slider, {
        props: {
          label: 'Single Value',
          value: 5,
          minimumValue: 1,
          maximumValue: 10,
          step: 1,
          fixedValues: [5],
          onValueChange: () => {},
        },
      });

      expect(screen.getByText('Single Value')).toBeInTheDocument();
    });
  });
});
