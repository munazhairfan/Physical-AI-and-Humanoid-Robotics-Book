/**
 * Locale Switching Guard for Docusaurus Docs Pages
 *
 * This script prevents 404 errors when switching locale from docs pages
 * by detecting locale switch events from /docs/* paths and redirecting
 * to appropriate locale-specific documentation paths.
 */

(function() {
  'use strict';

  // Check if we're on a docs page
  function isDocsPage() {
    return window.location.pathname.startsWith('/docs/');
  }

  // Get the current locale based on URL
  function getCurrentLocale() {
    const pathParts = window.location.pathname.split('/');
    // Check if the first part after domain is a locale code (e.g., 'en', 'es', 'fr')
    if (pathParts.length > 1 && pathParts[1] !== 'docs') {
      // It might be a locale prefix like /en/docs/, /es/docs/, etc.
      const potentialLocale = pathParts[1];
      // For now, we'll assume any non-docs first segment is a locale
      // In a real implementation, you might want to check against a known list
      return potentialLocale;
    }
    return null; // Default locale (no prefix)
  }

  // Generate the target locale-specific docs path
  function getTargetDocsPath(targetLocale) {
    const currentPath = window.location.pathname;
    const currentLocale = getCurrentLocale();

    if (currentLocale) {
      // Current path has locale prefix: /locale/docs/...
      // Remove the current locale prefix and add the target locale
      const pathWithoutLocale = currentPath.substring(currentLocale.length + 1); // +1 for the '/'
      if (targetLocale) {
        return '/' + targetLocale + pathWithoutLocale;
      } else {
        // Target is default locale (no prefix)
        return pathWithoutLocale;
      }
    } else {
      // Current path has no locale prefix (default locale)
      // Add the target locale prefix to the docs path
      if (targetLocale) {
        return '/' + targetLocale + currentPath;
      } else {
        // Already on default locale
        return currentPath;
      }
    }
  }

  // Enhanced version with better locale detection
  function isDocsPage() {
    const pathname = window.location.pathname;
    // Check if path starts with /docs/ or /{locale}/docs/
    return pathname.startsWith('/docs/') ||
           pathname.match(/^\/[a-zA-Z]{2}\/docs\//) ||
           pathname.match(/^\/[a-zA-Z]{2}-[a-zA-Z]{2}\/docs\//);
  }

  // Wait for the page to load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeLocaleGuard);
  } else {
    initializeLocaleGuard();
  }

  function initializeLocaleGuard() {
    // Monitor for locale switch events
    // Docusaurus typically uses links with data-locale attribute or similar
    // We'll intercept clicks on locale switching elements

    // Look for locale switcher elements (usually in navbar or footer)
    const localeSwitchers = document.querySelectorAll('[href*="/"][href*="/locale/"], [href*="/"][href^="/en"], [href*="/"][href^="/es"], [href*="/"][href^="/fr"], [href*="/"][href^="/de"], [href*="/"][href^="/ja"], [href*="/"][href^="/ko"], [href*="/"][href^="/zh"], .navbar__link[href^="/"]');

    localeSwitchers.forEach(switcher => {
      switcher.addEventListener('click', handleLocaleSwitch);
    });

    // Also monitor for URL changes that might indicate locale switching
    let currentPath = window.location.pathname;
    setInterval(() => {
      if (window.location.pathname !== currentPath) {
        currentPath = window.location.pathname;
        // Check if this was a locale switch from a docs page
        if (currentPath.startsWith('/docs/')) {
          // If we end up on /docs/... without a locale prefix after a switch,
          // we might need to redirect to appropriate locale
        }
      }
    }, 500);
  }

  function handleLocaleSwitch(event) {
    // Only handle locale switching from docs pages
    if (!isDocsPage()) {
      return; // Let Docusaurus handle normally
    }

    // Prevent default behavior to handle it ourselves
    event.preventDefault();

    // Get the target locale from the clicked element
    const targetUrl = event.currentTarget.getAttribute('href');
    if (!targetUrl) return;

    const targetPathname = new URL(targetUrl, window.location.origin).pathname;
    const targetLocale = extractLocaleFromPath(targetPathname);

    // Generate the correct docs path for the target locale
    const targetDocsPath = getTargetDocsPath(targetLocale);

    // Redirect to the locale-specific docs path
    window.location.href = targetDocsPath;
  }

  // Extract locale from a path like /en/docs/... or /es/docs/...
  function extractLocaleFromPath(pathname) {
    const pathParts = pathname.split('/');
    if (pathParts.length > 1) {
      const potentialLocale = pathParts[1];
      // Check if it looks like a locale (2-3 letters, possibly with region)
      if (potentialLocale.match(/^[a-zA-Z]{2}(-[a-zA-Z]{2})?$/)) {
        return potentialLocale;
      }
    }
    return null; // Default locale
  }

  // Alternative approach: Override the locale switching mechanism entirely
  // This looks for locale switcher dropdowns and intercepts their changes
  function setupLocaleSwitchIntercept() {
    // Wait a bit for Docusaurus to initialize
    setTimeout(() => {
      // Look for locale switcher dropdown or links
      const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
          mutation.addedNodes.forEach(function(node) {
            if (node.nodeType === 1) { // Element node
              // Check if this is a locale switcher element
              if (isLocaleSwitcherElement(node)) {
                attachLocaleSwitchHandler(node);
              }

              // Also check child elements
              const switchers = node.querySelectorAll ?
                node.querySelectorAll('[href*="/"][href^="/"], .locale-dropdown, .navbar__link[href^="/"]') :
                [];
              switchers.forEach(attachLocaleSwitchHandler);
            }
          });
        });
      });

      observer.observe(document.body, {
        childList: true,
        subtree: true
      });

      // Also check for existing locale switchers immediately
      const existingSwitchers = document.querySelectorAll(
        '.navbar__link, .dropdown__menu a, [data-locale], [href^="/en"], [href^="/es"], [href^="/fr"], [href^="/de"], [href^="/ja"], [href^="/ko"], [href^="/zh"]'
      );
      existingSwitchers.forEach(attachLocaleSwitchHandler);
    }, 1000);
  }

  function isLocaleSwitcherElement(element) {
    const href = element.getAttribute?.('href');
    const className = element.className || '';
    const textContent = element.textContent || '';

    // Check if it's a locale-related element
    return href?.startsWith('/en/') ||
           href?.startsWith('/es/') ||
           href?.startsWith('/fr/') ||
           href?.startsWith('/de/') ||
           href?.startsWith('/ja/') ||
           href?.startsWith('/ko/') ||
           href?.startsWith('/zh/') ||
           className.includes('locale') ||
           textContent.toLowerCase().includes('locale') ||
           textContent.toLowerCase().includes('language');
  }

  function attachLocaleSwitchHandler(element) {
    if (element._localeGuardAttached) return; // Prevent duplicate handlers

    element.addEventListener('click', function(event) {
      if (isDocsPage()) {
        event.preventDefault();

        const targetHref = element.getAttribute('href');
        if (targetHref) {
          const targetLocale = extractLocaleFromPath(targetHref);
          const targetDocsPath = getTargetDocsPath(targetLocale);
          window.location.href = targetDocsPath;
        }
      }
    });

    element._localeGuardAttached = true;
  }

  // Initialize the locale switch intercept
  setupLocaleSwitchIntercept();

})();