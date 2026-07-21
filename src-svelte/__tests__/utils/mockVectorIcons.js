/**
 * Framework-neutral mock for vector icons used in tests.
 *
 * The Svelte test tree does not depend on React, so icons are represented as
 * lightweight element descriptors with the same attributes and text content
 * as the original React mock.
 */

const createMockIcon = (name) => {
  const MockIcon = (props = {}) => {
    const { children: _children, ...iconProps } = props;

    return {
      type: 'i',
      props: {
        'data-testid': `icon-${name}`,
        'data-icon-name': props.name,
        ...iconProps,
      },
      children: props.name || name,
    };
  };

  MockIcon.displayName = `MockIcon(${name})`;
  return MockIcon;
};

export default {
  MaterialIcons: createMockIcon('MaterialIcons'),
  FontAwesome: createMockIcon('FontAwesome'),
  Ionicons: createMockIcon('Ionicons'),
  AntDesign: createMockIcon('AntDesign'),
  Entypo: createMockIcon('Entypo'),
  EvilIcons: createMockIcon('EvilIcons'),
  Feather: createMockIcon('Feather'),
  FontAwesome5: createMockIcon('FontAwesome5'),
  Foundation: createMockIcon('Foundation'),
  MaterialCommunityIcons: createMockIcon('MaterialCommunityIcons'),
  Octicons: createMockIcon('Octicons'),
  SimpleLineIcons: createMockIcon('SimpleLineIcons'),
  Zocial: createMockIcon('Zocial'),
};
