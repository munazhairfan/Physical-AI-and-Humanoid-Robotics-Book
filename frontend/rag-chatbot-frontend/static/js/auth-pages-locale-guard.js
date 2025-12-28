/**
 * Prevent locale switching on authentication pages and ensure proper navigation
 * This script ensures that login and signup pages remain in the default language
 * and handles navigation from locale-prefixed pages to auth pages
 */

(function() {
  'use strict';

  // Define supported locales
  const SUPPORTED_LOCALES = ['en', 'ur'];
  const DEFAULT_LOCALE = 'en';

  // Check if we're on an auth page
  function isAuthPage() {
    const pathname = window.location.pathname;
    return pathname === '/login' || pathname === '/signup' || pathname.startsWith('/auth/');
  }

  // Get the current locale based on URL
  function getCurrentLocale() {
    const pathname = window.location.pathname;
    const pathParts = pathname.split('/').filter(part => part !== '');

    // Check if the first part is a supported locale code
    if (pathParts.length > 0 && SUPPORTED_LOCALES.includes(pathParts[0])) {
      return pathParts[0];
    }
    return DEFAULT_LOCALE; // Default locale (no prefix)
  }

  // If we're on an auth page, intercept locale switcher clicks
  if (isAuthPage()) {
    // Listen for clicks on locale switcher elements
    document.addEventListener('click', function(event) {
      let targetElement = event.target;
      while (targetElement && targetElement !== document) {
        // Check if this is a locale dropdown item
        const href = targetElement.getAttribute?.('href');
        if (href) {
          const localeMatch = href.match(/^\/([a-zA-Z-]{2,5})(\/.*)?$/);
          if (localeMatch) {
            const targetLocale = localeMatch[1];

            // Check if this is a supported locale
            if (SUPPORTED_LOCALES.includes(targetLocale)) {
              // Prevent the default behavior on auth pages
              event.preventDefault();

              // Redirect to homepage in the target locale
              const homepageUrl = targetLocale === DEFAULT_LOCALE ? '/' : '/' + targetLocale + '/';
              window.location.href = homepageUrl;
              return;
            }
          }
        }

        targetElement = targetElement.parentElement;
      }
    }, true);
  }

  // If we're on a locale-prefixed page (like /ur/docs/*), ensure auth links work properly
  const currentLocale = getCurrentLocale();
  if (currentLocale !== DEFAULT_LOCALE) {
    // Listen for clicks on auth links (login, signup, etc.)
    document.addEventListener('click', function(event) {
      let targetElement = event.target;
      while (targetElement && targetElement !== document) {
        // Check if this is an auth link
        const href = targetElement.getAttribute?.('href');
        if (href) {
          // If it's an auth page link, ensure it goes to the default locale version
          if (href === '/login' || href === '/signup' || href.startsWith('/auth/')) {
            // Change the href to remove any current locale prefix
            event.preventDefault();
            // Always go to the default locale for auth pages
            window.location.href = href;
            return;
          }
        }

        targetElement = targetElement.parentElement;
      }
    }, true);
  }

})();